import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Portfolio360 v13 - Professional Portfolio Analytics", layout="wide")
st.title("📊 Portfolio360 v13 - Professional Portfolio Analytics & Forecasting")

# ----------- Sidebar Settings -------------
st.sidebar.header("🔧 Analysis Settings")

# Date range
date_start = st.sidebar.date_input("Analysis Start Date", value=datetime(2022, 1, 1))
date_end = st.sidebar.date_input("Analysis End Date", value=datetime.today())
if date_end < date_start:
    st.sidebar.error("End date must be after start date.")

# Number of simulated portfolios
n_portfolios = st.sidebar.slider("Number of Simulated Portfolios", 1000, 20000, 5000, 1000)

# Portfolio risk range
st.sidebar.markdown("---")
st.sidebar.subheader("Portfolio Risk Range")
min_risk = st.sidebar.selectbox("Minimum Portfolio Risk (%)", [i for i in range(0, 100, 10)], index=1)
max_risk = st.sidebar.selectbox("Maximum Portfolio Risk (%)", [i for i in range(10, 110, 10)], index=9)
st.sidebar.markdown(
    "<small>This range filters portfolios by risk (standard deviation). Only portfolios within this risk range will be displayed below.</small>",
    unsafe_allow_html=True
)

# Special parameters for each style
st.sidebar.markdown("---")
st.sidebar.subheader("Method-Specific Parameters")
target_return = st.sidebar.number_input("Target Return for Sortino and Omega (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.1) / 100
cvar_alpha = st.sidebar.slider("CVaR / VaR Confidence Level", 0.80, 0.99, 0.95, 0.01)

# Data upload
st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files for each asset (must have Date column and a price column)", 
    type=['csv'], 
    accept_multiple_files=True
)

# Price columns accepted
price_columns_possible = [
    "Adj Close", "adj close", "AdjClose", "adjclose",
    "Close", "close", "Last", "last", "Price", "price",
    "Close Price", "close price", "End", "end", "پایانی", "قیمت پایانی"
]

prices_df = pd.DataFrame()
asset_names = []
weight_settings = {}

if uploaded_files:
    for file in uploaded_files:
        name = file.name.split('.')[0]
        df = pd.read_csv(file)
        price_col = None
        for col in price_columns_possible:
            if col in df.columns:
                price_col = col
                break
        if price_col is None:
            st.error(
                f"File {file.name} does not have a recognized price column!"
                f"\nColumns: {list(df.columns)}"
                f"\nExpected: {price_columns_possible}"
            )
            continue
        if "Date" not in df.columns:
            st.error(f"File {file.name} must have a Date column! Columns: {list(df.columns)}")
            continue
        df = df[["Date", price_col]].rename(columns={price_col: name})
        df["Date"] = pd.to_datetime(df["Date"])
        df[name] = (
            df[name]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("٫", ".", regex=False)
            .str.replace("،", "", regex=False)
        )
        df[name] = pd.to_numeric(df[name], errors='coerce')
        df = df.dropna(subset=[name, "Date"])
        df = df.set_index("Date")
        df = df[(df.index >= pd.to_datetime(date_start)) & (df.index <= pd.to_datetime(date_end))]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)
    prices_df.dropna(inplace=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Asset Weight Constraints (%)")
    for name in asset_names:
        min_w = st.sidebar.number_input(f"Min Weight {name}", 0.0, 100.0, 0.0, 1.0, key=f"min_{name}") / 100
        max_w = st.sidebar.number_input(f"Max Weight {name}", 0.0, 100.0, 100.0, 1.0, key=f"max_{name}") / 100
        weight_settings[name] = {'min': min_w, 'max': max_w}

# --- Portfolio Analysis and Results
if not prices_df.empty:
    st.markdown("### Price Data (Preview)")
    st.dataframe(prices_df.tail())

    # Determine frequency for annualization
    freq = pd.infer_freq(prices_df.index)
    factor = 252
    if freq is not None:
        if freq[0].lower() == "m":
            factor = 12
        elif freq[0].lower() == "w":
            factor = 52

    returns = prices_df.pct_change().dropna()
    mean_ret = returns.mean() * factor
    cov = returns.cov() * factor

    def portfolio_return(weights):
        return np.dot(weights, mean_ret)
    def portfolio_risk(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

    min_risk_user = min_risk / 100
    max_risk_user = max_risk / 100

    styles = [
        ("Sharpe", "Portfolio with the highest Sharpe ratio. Classic risk-return optimization."),
        ("Sortino", "Portfolio with the highest Sortino ratio (return relative to downside risk below target)."),
        ("Omega", "Portfolio with the highest Omega ratio (reward to risk)."),
        ("CVaR", f"Portfolio with the lowest CVaR (Conditional Value-at-Risk) at {int(cvar_alpha*100)}% confidence level."),
        ("VaR", f"Portfolio with the lowest VaR (Value-at-Risk) at {int(cvar_alpha*100)}% confidence level."),
        ("Max Drawdown", "Portfolio with the lowest maximum drawdown."),
        ("Monte Carlo", "Optimal portfolio found by Monte Carlo simulation.")
    ]

    all_results = {}
    all_best = []

    st.markdown("### Portfolio Risk Distribution (All Methods)")
    results_for_hist = []
    for _ in range(n_portfolios):
        while True:
            w = np.random.dirichlet(np.ones(len(asset_names)))
            legal = True
            for i, n in enumerate(asset_names):
                if not (weight_settings[n]['min'] <= w[i] <= weight_settings[n]['max']):
                    legal = False
            if legal:
                break
        port_risk = portfolio_risk(w)
        results_for_hist.append(port_risk*100)
    # Professional English histogram for risk distribution
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(7,5))
    ax.hist(results_for_hist, bins=30, color='#4da6ff', edgecolor='black', alpha=0.85)
    ax.set_xlabel('Portfolio Risk (%)', fontsize=14)
    ax.set_ylabel('Number of Portfolios', fontsize=14)
    ax.set_title('Portfolio Risk Distribution (All Simulations)', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)

    for style, style_desc in styles:
        results = []
        for _ in range(n_portfolios):
            while True:
                w = np.random.dirichlet(np.ones(len(asset_names)))
                legal = True
                for i, n in enumerate(asset_names):
                    if not (weight_settings[n]['min'] <= w[i] <= weight_settings[n]['max']):
                        legal = False
                if legal:
                    break
            port_ret = portfolio_return(w)
            port_risk = portfolio_risk(w)
            port_sorted = np.dot(returns.values, w)
            downside = port_sorted[port_sorted < target_return]
            sortino = (port_ret - target_return) / (np.std(downside) + 1e-8) if len(downside) > 0 else 0
            omega = np.sum(port_sorted > target_return) / (np.abs(np.sum(port_sorted < target_return)) + 1e-8)
            cvar = np.mean(port_sorted[port_sorted <= np.percentile(port_sorted, (1-cvar_alpha)*100)]) if len(port_sorted) > 0 else 0
            var = np.percentile(port_sorted, (1-cvar_alpha)*100) if len(port_sorted) > 0 else 0
            cum = (1 + port_sorted).cumprod()
            peak = np.maximum.accumulate(cum)
            drawdowns = (cum - peak) / peak
            max_dd = drawdowns.min()
            sharpe = port_ret / (port_risk + 1e-8)
            results.append({
                "weights": w, "return": port_ret, "risk": port_risk, "sortino": sortino,
                "omega": omega, "cvar": cvar, "var": var, "drawdown": max_dd, "sharpe": sharpe
            })
        df_res = pd.DataFrame(results)
        # Show risk range info
        all_risks = df_res['risk']*100
        st.info(f"{style}: Min risk = {all_risks.min():.2f}% | Max risk = {all_risks.max():.2f}%")
        # Filter by user risk range
        df_res = df_res[(df_res["risk"] >= min_risk_user) & (df_res["risk"] <= max_risk_user)]
        if df_res.empty:
            st.warning(
                f"No portfolios found in risk range for {style}!\n\n"
                "Tips to fix this:\n"
                "- Use a wider risk range (e.g. Min 0%, Max 100%)\n"
                "- Increase the number of simulated portfolios\n"
                "- Loosen asset weight constraints\n"
                "- Check your price data for volatility\n"
                "- See the risk distribution histogram above for guidance."
            )
            continue
        # Best portfolio selection
        if style == "Sharpe":
            best = df_res.loc[df_res["sharpe"].idxmax()]
            best_desc = "Max Sharpe Ratio"
        elif style == "Sortino":
            best = df_res.loc[df_res["sortino"].idxmax()]
            best_desc = "Max Sortino Ratio"
        elif style == "Omega":
            best = df_res.loc[df_res["omega"].idxmax()]
            best_desc = "Max Omega Ratio"
        elif style == "CVaR":
            best = df_res.loc[df_res["cvar"].idxmin()]
            best_desc = "Min CVaR"
        elif style == "VaR":
            best = df_res.loc[df_res["var"].idxmin()]
            best_desc = "Min VaR"
        elif style == "Max Drawdown":
            best = df_res.loc[df_res["drawdown"].idxmax()]
            best_desc = "Min Max Drawdown"
        elif style == "Monte Carlo":
            best = df_res.loc[df_res["sharpe"].idxmax()]
            best_desc = "Best Monte Carlo (Max Sharpe)"
        all_results[style] = (df_res, best)
        if best["drawdown"] >= -0.3:
            all_best.append((style, best, best_desc))
        st.markdown(f"---\n### {style}: {best_desc}")
        st.markdown(f"**{style_desc}**")
        if style in ["Sortino", "Omega"]:
            st.info(f"Target Return: {target_return*100:.2f}%")
        if style in ["CVaR", "VaR"]:
            st.info(f"Confidence Level: {int(cvar_alpha*100)}%")
        # Professional Plotly Efficient Frontier
        fig2 = px.scatter(
            df_res, x=df_res["risk"]*100, y=df_res["return"]*100, color="sharpe",
            labels={'risk': 'Portfolio Risk (%)', 'return': 'Portfolio Return (%)', 'sharpe': 'Sharpe Ratio'},
            title=f"Efficient Frontier ({style})",
            color_continuous_scale="Viridis"
        )
        fig2.add_trace(go.Scatter(
            x=[best["risk"]*100], y=[best["return"]*100], mode="markers+text",
            marker=dict(size=14, color="red"), text=["Best Portfolio"], textposition="top center"
        ))
        fig2.update_layout(font=dict(family="DejaVu Sans", size=14))
        st.plotly_chart(fig2, use_container_width=True)
        # Portfolio weights table
        st.markdown("#### Portfolio Weights (Best Portfolio)")
        dfw = pd.DataFrame({'Asset': asset_names, 'Weight (%)': np.round(best['weights']*100, 2)})
        st.dataframe(dfw.set_index('Asset'), use_container_width=True)
        # Portfolio stats
        st.markdown(f"""
        - **Annual Return:** {best['return']*100:.2f}%
        - **Annual Risk:** {best['risk']*100:.2f}%
        - **Sharpe Ratio:** {best['sharpe']:.2f}
        - **Sortino Ratio:** {best['sortino']:.2f}
        - **Omega Ratio:** {best['omega']:.2f}
        - **CVaR ({int(cvar_alpha*100)}%):** {best['cvar']*100:.2f}%
        - **VaR ({int(cvar_alpha*100)}%):** {best['var']*100:.2f}%
        - **Max Drawdown:** {best['drawdown']*100:.2f}%
        """)
        st.info(
            f"**How to use {style}:** {style_desc}\n"
            "If you see a warning for this method, try expanding your risk range, loosening asset constraints, or increasing simulation count.",
            icon="ℹ️"
        )

    # Price Forecast for Each Asset (English, Professional)
    st.markdown("---\n## 🤖 Price Forecast for Each Asset (ARIMA Model)")
    pred_periods = st.slider("Forecast Periods (per asset)", 3, 24, 12, 1)
    for name in asset_names:
        ts = prices_df[name]
        try:
            model = ARIMA(ts, order=(1,1,1)).fit()
            pred = model.forecast(steps=pred_periods)
            df_pred = pd.concat([ts, pred])
            fig_pred = px.line(
                df_pred, 
                title=f"Price Forecast: {name}",
                labels={"value": "Price", "index": "Date"}
            )
            fig_pred.add_vline(x=ts.index[-1], line_dash="dot", line_color="red")
            fig_pred.update_layout(font=dict(family="DejaVu Sans", size=14))
            st.plotly_chart(fig_pred, use_container_width=True)
        except Exception as e:
            st.warning(f"Forecast for {name} failed! ({e})")

    # Best overall portfolio (risk < 30% drawdown, max return, min risk)
    st.markdown("---\n## 🏆 Best Portfolio (Risk ≤ 30% Drawdown)")
    if all_best:
        best_tuple = max(all_best, key=lambda x: (x[1]["return"], -x[1]["risk"]))
        style, best, desc = best_tuple
        st.success(f"Suggested Portfolio: Style '{style}' ({desc})")
        dfw = pd.DataFrame({'Asset': asset_names, 'Weight (%)': np.round(best['weights']*100, 2)})
        st.dataframe(dfw.set_index('Asset'), use_container_width=True)
        st.markdown(f"""
        - **Annual Return:** {best['return']*100:.2f}%
        - **Annual Risk:** {best['risk']*100:.2f}%
        - **Sharpe Ratio:** {best['sharpe']:.2f}
        - **Sortino Ratio:** {best['sortino']:.2f}
        - **Omega Ratio:** {best['omega']:.2f}
        - **CVaR ({int(cvar_alpha*100)}%):** {best['cvar']*100:.2f}%
        - **VaR ({int(cvar_alpha*100)}%):** {best['var']*100:.2f}%
        - **Max Drawdown:** {best['drawdown']*100:.2f}%
        """)
    else:
        st.warning("No portfolio found with drawdown ≤ 30%!")

    st.markdown("""
    <div style="text-align:left;">
    <b>App Guide:</b><br>
    - All portfolio optimization styles are calculated simultaneously.<br>
    - For each style: efficient frontier, key parameters, optimal portfolio, and full explanation are provided.<br>
    - Price forecast for each asset using ARIMA is available.<br>
    - The best overall portfolio with max return and risk ≤ 30% drawdown is suggested.<br>
    - Change asset weight constraints, risk range, and style parameters as needed for your scenario.<br>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Please upload price data for your assets to start analysis.")
