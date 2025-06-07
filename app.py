import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="تحلیل پرتفو و پیش‌بینی", layout="wide")
st.title("📊 Portfolio360 - ابزار تحلیل پرتفو و استراتژی Married Put")

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
    mean_ret = returns.mean() * factor_annual
    cov = returns.cov() * factor_annual

    # ۱. پارامترهای هر دارایی (Married Put + بیمه)
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

    # ۲. استراتژی Married Put برای هر دارایی (نمودار و جدول)
    st.markdown("## 💼 Married Put Strategy & Insurance Simulation")
    for asset in asset_names:
        p = asset_settings[asset]
        st.markdown(f"### {asset} - Married Put Strategy")
        prices = np.linspace(p["spot"]*0.5, p["spot"]*1.5, 200)
        profit_stock = (prices - p["spot"]) * p["qty"]
        profit_put = np.maximum(p["strike"] - prices, 0) * p["contract_size"] - p["premium"] * p["contract_size"]
        profit_strategy = profit_stock + profit_put

        # نمودار P/L با استایل مشابه نمونه کاربر
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

    # ۳. محاسبه ریسک ماهانه و سالانه هر دارایی و بیمه
    st.markdown("## 📊 Risk Calculation (Monthly & Annualized)")
    risk_table = []
    for asset in asset_names:
        ret = returns[asset]
        monthly_std = np.std(ret) * np.sqrt(factor_monthly)
        annual_std = np.std(ret) * np.sqrt(factor_annual)
        insured = asset_settings[asset]["insured"]
        # اگر بیمه شده، ریسک را کاهش بده (مثلاً نصف کن)
        risk_final = annual_std * 0.5 if insured else annual_std
        risk_table.append({
            "Asset": asset,
            "Monthly Risk (%)": round(monthly_std*100, 2),
            "Annual Risk (%)": round(annual_std*100, 2),
            "Insured": "Yes" if insured else "No",
            "Effective Annual Risk (%)": round(risk_final*100, 2)
        })
    st.dataframe(pd.DataFrame(risk_table).set_index("Asset"), use_container_width=True)

    # ۴. تحلیل پرتفو با لحاظ بیمه و وزن‌دهی ویژه برای دارایی‌های بیمه‌شده و پرریسک
    def portfolio_return(weights):
        return np.dot(weights, mean_ret)
    def portfolio_risk(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    n_portfolios = 5000
    min_risk_user = 0.0
    max_risk_user = 2.0

    # وزن‌دهی ویژه: اگر بیمه فعال باشد و ریسک سالانه بالاتر از میانه باشد، min وزن به جای 0، 0.1 بگذار!
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
    all_best = []

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
            port_ret = portfolio_return(w)
            port_risk_annual = portfolio_risk(w)
            port_risk_monthly = port_risk_annual / np.sqrt(factor_annual/factor_monthly)
            port_sorted = np.dot(returns.values, w)
            target_return = 0.05
            downside = port_sorted[port_sorted < target_return]
            sortino = (port_ret - target_return) / (np.std(downside) + 1e-8) if len(downside) > 0 else 0
            omega = np.sum(port_sorted > target_return) / (np.abs(np.sum(port_sorted < target_return)) + 1e-8)
            cvar = np.mean(port_sorted[port_sorted <= np.percentile(port_sorted, (1-cvar_alpha)*100)]) if len(port_sorted) > 0 else 0
            var = np.percentile(port_sorted, (1-cvar_alpha)*100) if len(port_sorted) > 0 else 0
            cum = (1 + port_sorted).cumprod()
            peak = np.maximum.accumulate(cum)
            drawdowns = (cum - peak) / peak
            max_dd = drawdowns.min()
            sharpe = port_ret / (port_risk_annual + 1e-8)
            results.append({
                "weights": w, "return": port_ret,
                "risk_annual": port_risk_annual, "risk_monthly": port_risk_monthly,
                "sortino": sortino, "omega": omega,
                "cvar": cvar, "var": var,
                "drawdown": max_dd, "sharpe": sharpe
            })
        df_res = pd.DataFrame(results)
        df_res = df_res[(df_res["risk_annual"] >= min_risk_user) & (df_res["risk_annual"] <= max_risk_user)]

        st.markdown(f"---\n### {style} : {style_desc}")
        if style in ["CVaR", "VaR"]:
            st.info(f"سطح اطمینان: {int(cvar_alpha*100)}٪")

        if df_res.empty:
            st.warning(f"هیچ سبدی برای این سبک در محدوده ریسک پیدا نشد!")
            fig2 = px.scatter(
                x=[], y=[],
                labels={'risk_annual': 'Annual Risk (%)', 'return': 'Annual Return (%)', 'sharpe': 'Sharpe Ratio'},
                title=f"Efficient Frontier ({style})"
            )
            fig2.update_layout(font=dict(family="DejaVu Sans", size=14))
            st.plotly_chart(fig2, use_container_width=True)
            continue

        if style == "Sharpe":
            best = df_res.loc[df_res["sharpe"].idxmax()]
            best_desc = "بیشترین نسبت شارپ"
        elif style == "Sortino":
            best = df_res.loc[df_res["sortino"].idxmax()]
            best_desc = "بیشترین نسبت سورتینو"
        elif style == "Omega":
            best = df_res.loc[df_res["omega"].idxmax()]
            best_desc = "بیشترین نسبت امگا"
        elif style == "CVaR":
            best = df_res.loc[df_res["cvar"].idxmin()]
            best_desc = "کمترین CVaR"
        elif style == "VaR":
            best = df_res.loc[df_res["var"].idxmin()]
            best_desc = "کمترین VaR"
        elif style == "Max Drawdown":
            best = df_res.loc[df_res["drawdown"].idxmax()]
            best_desc = "کمترین افت سرمایه"
        elif style == "Monte Carlo":
            best = df_res.loc[df_res["sharpe"].idxmax()]
            best_desc = "بهترین مونت کارلو (شارپ ماکزیمم)"
        all_results[style] = (df_res, best)
        if best["drawdown"] >= -0.3:
            all_best.append((style, best, best_desc))

        fig2 = px.scatter(
            df_res, x=df_res["risk_annual"]*100, y=df_res["return"]*100, color="sharpe",
            labels={'risk_annual': 'Annual Risk (%)', 'return': 'Annual Return (%)', 'sharpe': 'Sharpe Ratio'},
            title=f"Efficient Frontier ({style})",
            color_continuous_scale="Viridis"
        )
        fig2.add_trace(go.Scatter(
            x=[best["risk_annual"]*100], y=[best["return"]*100], mode="markers+text",
            marker=dict(size=14, color="red"), text=["Best Portfolio"], textposition="top center"
        ))
        fig2.update_layout(font=dict(family="DejaVu Sans", size=14))
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### جدول وزن پرتفو (سبد بهینه)")
        dfw = pd.DataFrame({'دارایی': asset_names, 'وزن (%)': np.round(best['weights']*100, 4)})
        st.dataframe(dfw.set_index('دارایی'), use_container_width=True)
        st.markdown(f"""
        - **بازده سالانه:** {best['return']*100:.2f}%
        - **ریسک سالانه:** {best['risk_annual']*100:.2f}%
        - **ریسک ماهانه:** {best['risk_monthly']*100:.2f}%
        - **نسبت شارپ:** {best['sharpe']:.2f}
        - **سورتینو:** {best['sortino']:.2f}
        - **امگا:** {best['omega']:.2f}
        - **CVaR ({int(cvar_alpha*100)}%):** {best['cvar']*100:.2f}%
        - **VaR ({int(cvar_alpha*100)}%):** {best['var']*100:.2f}%
        - **Max Drawdown:** {best['drawdown']*100:.2f}%
        """)
        st.info(
            f"راهنمای استفاده از سبک {style}: {style_desc}\n"
            "در صورت مشاهده خطا، بازه ریسک یا پارامترهای کد را تغییر دهید."
        )

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

    st.markdown("---\n## 🏆 بهترین پرتفو با شرط افت سرمایه حداکثر 30٪")
    if all_best:
        best_tuple = max(all_best, key=lambda x: (x[1]["return"], -x[1]["risk_annual"]))
        style, best, desc = best_tuple
        st.success(f"سبد پیشنهادی: سبک '{style}' ({desc})")
        dfw = pd.DataFrame({'دارایی': asset_names, 'وزن (%)': np.round(best['weights']*100, 4)})
        st.dataframe(dfw.set_index('دارایی'), use_container_width=True)
        st.markdown(f"""
        - **بازده سالانه:** {best['return']*100:.2f}%
        - **ریسک سالانه:** {best['risk_annual']*100:.2f}%
        - **ریسک ماهانه:** {best['risk_monthly']*100:.2f}%
        - **نسبت شارپ:** {best['sharpe']:.2f}
        - **سورتینو:** {best['sortino']:.2f}
        - **امگا:** {best['omega']:.2f}
        - **CVaR ({int(cvar_alpha*100)}%):** {best['cvar']*100:.2f}%
        - **VaR ({int(cvar_alpha*100)}%):** {best['var']*100:.2f}%
        - **Max Drawdown:** {best['drawdown']*100:.2f}%
        """)
    else:
        st.warning("هیچ پرتفوئی با شرط افت سرمایه کمتر از ۳۰٪ پیدا نشد!")

    st.markdown("""
    <div dir="rtl" style="text-align:right;">
    <b>راهنما:</b><br>
    - برای هر دارایی می‌توانید بیمه و پارامترهای Married Put را تنظیم کنید.<br>
    - ریسک ماهانه و سالانه و اثر بیمه‌کردن دارایی محاسبه و در پرتفو لحاظ می‌شود.<br>
    - نمودارهای Married Put تم تیره و حرفه‌ای دارند.<br>
    - سبک‌های مختلف پرتفو همیشه نمایش داده می‌شوند.<br>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("لطفاً ابتدا داده قیمت دارایی‌ها را آپلود کنید.")
