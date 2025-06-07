import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="تحلیل پرتفو و Married Put", layout="wide")
st.title("📊 Portfolio360 - تحلیل پرتفو و استراتژی Married Put")

st.sidebar.markdown("---")
st.sidebar.subheader("پارامترهای تخصصی")
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR/VaR", 0.80, 0.99, 0.95, 0.01)

uploaded_files = st.sidebar.file_uploader(
    "آپلود فایل CSV دارایی‌ها (ستون Date و ستون قیمت)", 
    type=['csv'], 
    accept_multiple_files=True
)

prices_df = pd.DataFrame()
asset_names = []
asset_settings = {}

price_columns_possible = [
    "Adj Close", "adj close", "AdjClose", "adjclose",
    "Close", "close", "Last", "last", "Price", "price",
    "Close Price", "close price", "End", "end", "پایانی", "قیمت پایانی"
]

style_explanations = {
    "Sharpe": "سبد بهینه شارپ، بیشترین نسبت بازده به ریسک را دارد و برای سرمایه‌گذاران ریسک‌گریز مناسب است. این سبد مناسب زمانی است که می‌خواهید با ریسکی معقول، بیشترین بازدهی نسبی را کسب کنید.",
    "Sortino": "سبد سورتینو برای مواقعی مناسب است که فقط به ریسک منفی (ریسک سقوط) حساس هستید و بازده بالاتر از مقدار هدف برایتان اهمیت دارد.",
    "Omega": "سبد امگا مناسب برای کسانی است که می‌خواهند احتمال کسب بازدهی بالاتر از یک آستانه مطلوب را بیشینه کنند. این سبد برای سرمایه‌گذاران با نگاه مبتنی بر هدف مناسب است.",
    "CVaR": "سبد کمینه CVaR برای شرایط بحرانی که تمرکز روی کنترل ضررهای سنگین و ریسک‌های دم‌کلفت است مناسب است.",
    "VaR": "سبد کمینه VaR برای سرمایه‌گذارانی مناسب است که بیشترین افت احتمالی در یک سطح اطمینان خاص برایشان اهمیت دارد.",
    "Max Drawdown": "سبد کمینه افت سرمایه (Max Drawdown) برای کسانی کاربرد دارد که تحمل افت پیاپی و شدید ارزش پرتفو را ندارند.",
    "Monte Carlo": "سبد مونت‌کارلو بهترین سبد بین نمونه‌های تصادفی است (معمولاً بر اساس بیشترین شارپ)، مناسب برای سنجش و مقایسه سایر سبک‌ها."
}

if uploaded_files:
    for file in uploaded_files:
        name = file.name.split('.')[0]
        df = pd.read_csv(file)
        price_col = None
        for col in price_columns_possible:
            if col in df.columns:
                price_col = col
                break
        if price_col is None or "Date" not in df.columns:
            st.error(f"فایل {file.name} باید ستون Date و یک ستون قیمت معتبر داشته باشد! ستون‌ها: {list(df.columns)}")
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
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)
    prices_df.dropna(inplace=True)

if not prices_df.empty:
    st.markdown("### 📈 داده‌های قیمت پایانی دارایی‌ها")
    st.dataframe(prices_df.tail())

    freq = pd.infer_freq(prices_df.index)
    factor_annual = 252
    factor_monthly = 21
    if freq is not None:
        if freq[0].lower() == "m":
            factor_annual = 12
            factor_monthly = 1
        elif freq[0].lower() == "w":
            factor_annual = 52
            factor_monthly = 4

    returns = prices_df.pct_change().dropna()
    mean_ret_annual = returns.mean() * factor_annual
    mean_ret_monthly = returns.mean() * factor_monthly
    cov = returns.cov() * factor_annual

    # پارامترهای هر دارایی (Married Put + بیمه)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Married Put و بیمه برای هر دارایی")
    for asset in asset_names:
        st.sidebar.markdown(f"**{asset}**")
        base_price = float(prices_df[asset].iloc[-1])
        insured = st.sidebar.checkbox(f"بیمه دارایی {asset}", value=False, key=f"insured_{asset}")
        spot = st.sidebar.number_input(f"Spot Price ({asset})", value=base_price, format="%.6f", key=f"spot_{asset}")
        qty = st.sidebar.number_input(f"Quantity ({asset})", min_value=0.0, value=0.0002, format="%.6f", key=f"qty_{asset}")
        strike = st.sidebar.number_input(f"Strike Price ({asset})", value=base_price, format="%.6f", key=f"strike_{asset}")
        premium = st.sidebar.number_input(f"Put Premium ({asset})", min_value=0.0, value=10.0, format="%.6f", key=f"premium_{asset}")
        contract_size = st.sidebar.number_input(f"Contract Size ({asset})", min_value=0.0, value=1.0, format="%.6f", key=f"contract_{asset}")
        asset_settings[asset] = {
            "insured": insured, "spot": spot, "qty": float(f"{qty:.6f}"),
            "strike": strike, "premium": premium, "contract_size": float(f"{contract_size:.6f}")
        }

    # Married Put برای هر دارایی
    st.markdown("## 💼 Married Put Strategy & Insurance Simulation")
    for asset in asset_names:
        p = asset_settings[asset]
        st.markdown(f"### {asset} - Married Put Strategy")
        prices = np.linspace(p["spot"]*0.5, p["spot"]*1.5, 200)
        profit_stock = (prices - p["spot"]) * p["qty"]
        profit_put = np.maximum(p["strike"] - prices, 0) * p["contract_size"] - p["premium"] * p["contract_size"]
        profit_strategy = profit_stock + profit_put

        # درصد سود و ضرر
        max_profit = np.max(profit_strategy)
        min_profit = np.min(profit_strategy)
        investment = p["spot"] * p["qty"] if p["spot"] * p["qty"] != 0 else 1
        percent_profit = 100 * max_profit / investment
        percent_loss = 100 * min_profit / investment

        # نمودار P/L با annotation درصد سود و زیان
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices, y=profit_strategy,
            mode='lines',
            line=dict(color='white', width=2),
            name='Profit/Loss'
        ))
        fig.add_trace(go.Scatter(
            x=prices, y=np.zeros_like(prices),
            mode='lines', line=dict(color='gray', width=1, dash='dot'),
            showlegend=False
        ))
        fig.add_traces([
            go.Scatter(
                x=prices, y=np.where(profit_strategy < 0, profit_strategy, 0),
                fill='tozeroy', fillcolor='rgba(255,0,0,0.25)', line=dict(color='rgba(0,0,0,0)'), showlegend=False
            ),
            go.Scatter(
                x=prices, y=np.where(profit_strategy > 0, profit_strategy, 0),
                fill='tozeroy', fillcolor='rgba(0,255,0,0.18)', line=dict(color='rgba(0,0,0,0)'), showlegend=False
            ),
        ])
        fig.add_vline(x=p["spot"], line_color="gray", line_dash='dot')
        fig.add_annotation(
            x=p["spot"], y=profit_strategy[np.abs(prices-p["spot"]).argmin()],
            text=f"{asset} {p['qty']} @ {p['spot']:,.2f}", showarrow=False,
            font=dict(color='white'), bgcolor='#111'
        )
        # annotation درصد سود
        fig.add_annotation(
            x=prices[np.argmax(profit_strategy)],
            y=max_profit,
            text=f"+{percent_profit:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=40,
            ay=-40,
            font=dict(color='lightgreen', size=16),
            bgcolor='rgba(0,70,0,0.7)',
            bordercolor='green',
            borderwidth=2
        )
        # annotation درصد ضرر
        fig.add_annotation(
            x=prices[np.argmin(profit_strategy)],
            y=min_profit,
            text=f"{percent_loss:.1f}%",
            showarrow=True,
            arrowhead=2,
            ax=-40,
            ay=40,
            font=dict(color='pink', size=16),
            bgcolor='rgba(70,0,0,0.7)',
            bordercolor='red',
            borderwidth=2
        )
        fig.update_layout(
            template="plotly_dark",
            title="Expected Profit / Loss - Married Put",
            xaxis_title="Underlying Price",
            yaxis_title="Profit / Loss",
            font=dict(family="DejaVu Sans", size=14),
            margin=dict(l=30, r=30, t=50, b=30),
            plot_bgcolor="#181818",
            paper_bgcolor="#181818"
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        # جدول پارامترها و سود/زیان
        pl_table = pd.DataFrame({
            "Underlying Price": np.round(prices, 2),
            "Profit/Loss": np.round(profit_strategy, 2)
        })
        st.dataframe(pl_table.iloc[::40, :], use_container_width=True)

    st.markdown("## 📊 Risk & Return Calculation (Monthly & Annualized)")
    risk_table = []
    for asset in asset_names:
        ret = returns[asset]
        monthly_std = np.std(ret) * np.sqrt(factor_monthly)
        annual_std = np.std(ret) * np.sqrt(factor_annual)
        mean_monthly = np.mean(ret) * factor_monthly
        mean_annual = np.mean(ret) * factor_annual
        insured = asset_settings[asset]["insured"]
        effective_annual_risk = annual_std * 0.5 if insured else annual_std
        effective_annual_return = mean_annual - (asset_settings[asset]["premium"] * asset_settings[asset]["contract_size"] / prices_df[asset].iloc[-1] if insured else 0)
        risk_table.append({
            "Asset": asset,
            "Monthly Return (%)": round(mean_monthly*100, 2),
            "Annual Return (%)": round(mean_annual*100, 2),
            "Monthly Risk (%)": round(monthly_std*100, 2),
            "Annual Risk (%)": round(annual_std*100, 2),
            "Insured": "Yes" if insured else "No",
            "Effective Annual Risk (%)": round(effective_annual_risk*100, 2),
            "Effective Annual Return (%)": round(effective_annual_return*100, 2)
        })
    st.dataframe(pd.DataFrame(risk_table).set_index("Asset"), use_container_width=True)

    # تحلیل پرتفو با لحاظ بیمه و وزن‌دهی ویژه برای دارایی‌های بیمه‌شده و پرریسک
    def portfolio_return(weights, mean_ret):
        return np.dot(weights, mean_ret)
    def portfolio_risk(weights, cov):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    n_portfolios = 5000
    min_risk_user = 0.0
    max_risk_user = 2.0

    risks = pd.DataFrame(risk_table).set_index("Asset")["Annual Risk (%)"]
    median_risk = risks.median()
    for asset in asset_names:
        insured = asset_settings[asset]["insured"]
        high_risk = risks.loc[asset] > median_risk
        min_weight = 0.1 if insured and high_risk else 0.0
        asset_settings[asset]["min_weight"] = min_weight
        asset_settings[asset]["max_weight"] = 1.0

    styles = [
        ("Sharpe", "پرتفو با بیشترین نسبت شارپ."),
        ("Sortino", "پرتفو با بیشترین نسبت سورتینو."),
        ("Omega", "پرتفو با بیشترین نسبت امگا."),
        ("CVaR", f"پرتفو با کمترین CVaR در سطح اطمینان {int(cvar_alpha*100)}٪."),
        ("VaR", f"پرتفو با کمترین VaR در سطح اطمینان {int(cvar_alpha*100)}٪."),
        ("Max Drawdown", "پرتفو با کمترین افت سرمایه."),
        ("Monte Carlo", "پرتفو بهینه با شبیه‌سازی Monte Carlo.")
    ]
    all_results = {}
    best_overall = None
    best_score = -np.inf

    for style, style_desc in styles:
        results = []
        for _ in range(n_portfolios):
            while True:
                w = np.random.dirichlet(np.ones(len(asset_names)))
                legal = True
                for i, n in enumerate(asset_names):
                    mn = asset_settings[n]["min_weight"]
                    mx = asset_settings[n]["max_weight"]
                    if not (mn <= w[i] <= mx):
                        legal = False
                if legal:
                    break

            mean_ret_adj = []
            for i, n in enumerate(asset_names):
                insured = asset_settings[n]["insured"]
                if insured:
                    ret = mean_ret_annual[n] - (asset_settings[n]["premium"] * asset_settings[n]["contract_size"] / prices_df[n].iloc[-1])
                else:
                    ret = mean_ret_annual[n]
                mean_ret_adj.append(ret)
            mean_ret_adj = np.array(mean_ret_adj)

            cov_adj = cov.copy()
            for i, n in enumerate(asset_names):
                if asset_settings[n]["insured"]:
                    cov_adj.iloc[i, :] *= 0.5
                    cov_adj.iloc[:, i] *= 0.5

            port_ret_annual = portfolio_return(w, mean_ret_adj)
            port_ret_monthly = portfolio_return(w, mean_ret_monthly)
            port_risk_annual = portfolio_risk(w, cov_adj)
            port_risk_monthly = port_risk_annual / np.sqrt(factor_annual/factor_monthly)
            port_sorted = np.dot(returns.values, w)
            target_return = 0.05
            downside = port_sorted[port_sorted < target_return]
            sharpe = port_ret_annual / (port_risk_annual + 1e-8)
            sortino = (port_ret_annual - target_return) / (np.std(downside) + 1e-8) if len(downside) > 0 else 0
            omega = np.sum(port_sorted > target_return) / (np.abs(np.sum(port_sorted < target_return)) + 1e-8)
            cvar = np.mean(port_sorted[port_sorted <= np.percentile(port_sorted, (1-cvar_alpha)*100)]) if len(port_sorted) > 0 else 0
            var = np.percentile(port_sorted, (1-cvar_alpha)*100) if len(port_sorted) > 0 else 0
            cum = (1 + port_sorted).cumprod()
            peak = np.maximum.accumulate(cum)
            drawdowns = (cum - peak) / peak
            max_dd = drawdowns.min()
            results.append({
                "weights": w, "return_annual": port_ret_annual, "return_monthly": port_ret_monthly,
                "risk_annual": port_risk_annual, "risk_monthly": port_risk_monthly,
                "sortino": sortino, "omega": omega,
                "cvar": cvar, "var": var,
                "drawdown": max_dd, "sharpe": sharpe
            })
        df_res = pd.DataFrame(results)
        df_res = df_res[(df_res["risk_annual"] >= min_risk_user) & (df_res["risk_annual"] <= max_risk_user)]

        st.markdown(f"---\n### {style} : {style_desc}")
        st.info(style_explanations.get(style, ""))
        if style in ["CVaR", "VaR"]:
            st.info(f"سطح اطمینان: {int(cvar_alpha*100)}٪")
        if df_res.empty:
            st.warning(f"هیچ سبدی برای این سبک در محدوده ریسک پیدا نشد!")
            fig2 = px.scatter(
                x=[], y=[],
                labels={'risk_annual': 'Annual Risk (%)', 'return_annual': 'Annual Return (%)', 'sharpe': 'Sharpe Ratio'},
                title=f"Efficient Frontier ({style})"
            )
            fig2.update_layout(font=dict(family="DejaVu Sans", size=14))
            st.plotly_chart(fig2, use_container_width=True)
            continue

        if style == "Sharpe":
            best = df_res.loc[df_res["sharpe"].idxmax()]
        elif style == "Sortino":
            best = df_res.loc[df_res["sortino"].idxmax()]
        elif style == "Omega":
            best = df_res.loc[df_res["omega"].idxmax()]
        elif style == "CVaR":
            best = df_res.loc[df_res["cvar"].idxmin()]
        elif style == "VaR":
            best = df_res.loc[df_res["var"].idxmin()]
        elif style == "Max Drawdown":
            best = df_res.loc[df_res["drawdown"].idxmax()]
        elif style == "Monte Carlo":
            best = df_res.loc[df_res["sharpe"].idxmax()]
        all_results[style] = (df_res, best)

        # معیار انتخاب بهترین: بیشترین نسبت شارپ (بازده تقسیم بر ریسک)
        score = best['return_annual'] / (best['risk_annual'] + 1e-8)
        if score > best_score:
            best_score = score
            best_overall = (style, best)

        fig2 = px.scatter(
            df_res, x=df_res["risk_annual"]*100, y=df_res["return_annual"]*100, color="sharpe",
            labels={'risk_annual': 'Annual Risk (%)', 'return_annual': 'Annual Return (%)', 'sharpe': 'Sharpe Ratio'},
            title=f"Efficient Frontier ({style})",
            color_continuous_scale="Viridis"
        )
        fig2.add_trace(go.Scatter(
            x=[best["risk_annual"]*100], y=[best["return_annual"]*100], mode="markers+text",
            marker=dict(size=14, color="red"), text=["Best Portfolio"], textposition="top center"
        ))
        fig2.update_layout(font=dict(family="DejaVu Sans", size=14))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### جدول وزن پرتفو (سبد بهینه)")
        dfw = pd.DataFrame({'دارایی': asset_names, 'وزن (%)': np.round(best['weights']*100, 4)})
        st.dataframe(dfw.set_index('دارایی'), use_container_width=True)
        st.markdown(f"""
        - **بازده سالانه:** {best['return_annual']*100:.2f}%
        - **بازده ماهانه:** {best['return_monthly']*100:.2f}%
        - **ریسک سالانه:** {best['risk_annual']*100:.2f}%
        - **ریسک ماهانه:** {best['risk_monthly']*100:.2f}%
        - **نسبت شارپ:** {best['sharpe']:.2f}
        - **سورتینو:** {best['sortino']:.2f}
        - **امگا:** {best['omega']:.2f}
        - **CVaR ({int(cvar_alpha*100)}%):** {best['cvar']*100:.2f}%
        - **VaR ({int(cvar_alpha*100)}%):** {best['var']*100:.2f}%
        - **Max Drawdown:** {best['drawdown']*100:.2f}%
        """)

    # پیشنهاد بهترین سبک بر اساس کمترین ریسک و بیشترین بازده (Sharpe-like)
    if best_overall:
        style, best = best_overall
        st.success(f"📢 بهترین سبد پیشنهادی بین همه سبک‌ها: سبک **{style}**")
        st.markdown(f"""
        - **بازده سالانه:** {best['return_annual']*100:.2f}%
        - **ریسک سالانه:** {best['risk_annual']*100:.2f}%
        - **نسبت شارپ:** {best['sharpe']:.2f}

        <div dir="rtl" style="text-align:right; font-size:16px;">
        این سبک به دلیل ترکیب مناسب بیشترین بازده سالانه و کمترین ریسک سالانه (نسبت شارپ بالاتر) به عنوان بهترین گزینه معرفی شد. این انتخاب مناسب کسانی است که می‌خواهند در کنار کنترل ریسک، بازدهی بالاتری نیز داشته باشند.
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("لطفاً ابتدا داده قیمت دارایی‌ها را آپلود کنید.")