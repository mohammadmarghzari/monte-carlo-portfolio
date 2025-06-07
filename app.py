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

st.set_page_config(page_title="Portfolio360 - تحلیل پرتفو و پیش‌بینی", layout="wide")
st.title("📊 Portfolio360 - ابزار تحلیل پرتفو و پیش‌بینی قیمت")

# ----------- سایدبار تنظیمات -------------
st.sidebar.header("🔧 تنظیمات تحلیل")

# تاریخ شروع و پایان
date_start = st.sidebar.date_input("تاریخ شروع تحلیل", value=datetime(2022, 1, 1))
date_end = st.sidebar.date_input("تاریخ پایان تحلیل", value=datetime.today())
if date_end < date_start:
    st.sidebar.error("تاریخ پایان باید بعد از تاریخ شروع باشد.")

# تعداد پرتفوهای شبیه‌سازی
n_portfolios = st.sidebar.slider("تعداد پرتفوهای شبیه‌سازی", 1000, 20000, 5000, 1000)

# پارامترهای ریسک پرتفو
st.sidebar.markdown("---")
st.sidebar.subheader("محدوده ریسک پرتفو")
min_risk = st.sidebar.selectbox("حداقل ریسک پرتفو (%)", [i for i in range(0, 100, 10)], index=1)
max_risk = st.sidebar.selectbox("حداکثر ریسک پرتفو (%)", [i for i in range(10, 110, 10)], index=9)
st.sidebar.markdown(
    "<small>این فیلدها محدوده ریسک پرتفوها را روی مرز کارا تعیین می‌کنند و فقط سبدهایی با ریسک در این بازه نمایش داده می‌شوند.</small>",
    unsafe_allow_html=True
)

# پارامترهای ویژه هر سبک
st.sidebar.markdown("---")
st.sidebar.subheader("پارامترهای روش‌ها")
target_return = st.sidebar.number_input("بازده هدف برای سورتینو و امگا (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.1) / 100
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR/VaR", 0.80, 0.99, 0.95, 0.01)

# بارگذاری داده‌ها
st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader(
    "آپلود فایل CSV قیمت دارایی‌ها (ستون تاریخ و قیمت - هر نام متداول برای قیمت پایانی)", 
    type=['csv'], 
    accept_multiple_files=True
)

# پردازش فایل‌ها
prices_df = pd.DataFrame()
asset_names = []
weight_settings = {}

# اسامی متداول ستون قیمت بسته
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
        if price_col is None:
            st.error(
                f"فایل {file.name} هیچ‌کدام از ستون‌های متداول قیمت پایانی را ندارد!"
                f"\nستون‌های فعلی فایل: {list(df.columns)}"
                f"\nستون‌های مورد انتظار: {price_columns_possible}"
            )
            continue
        if "Date" not in df.columns:
            st.error(f"فایل {file.name} باید دارای ستون تاریخ (Date) باشد! ستون‌های فعلی: {list(df.columns)}")
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
    st.sidebar.subheader("محدودیت وزن هر دارایی (٪)")
    for name in asset_names:
        min_w = st.sidebar.number_input(f"حداقل وزن {name}", 0.0, 100.0, 0.0, 1.0, key=f"min_{name}") / 100
        max_w = st.sidebar.number_input(f"حداکثر وزن {name}", 0.0, 100.0, 100.0, 1.0, key=f"max_{name}") / 100
        weight_settings[name] = {'min': min_w, 'max': max_w}

# تحلیل پرتفو و نمایش نتایج
if not prices_df.empty:
    st.markdown("### 📈 داده‌های قیمت پایانی تعدیل‌شده دارایی‌ها")
    st.dataframe(prices_df.tail())

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
        df_res = df_res[(df_res["risk"] >= min_risk_user) & (df_res["risk"] <= max_risk_user)]
        if df_res.empty:
            st.warning(
                f"در سبک {style} هیچ سبدی در محدوده ریسک انتخابی پیدا نشد! "
                "\n\n"
                "- بازه ریسک را بزرگ‌تر انتخاب کنید.\n"
                "- تعداد پرتفوهای شبیه‌سازی را افزایش دهید.\n"
                "- محدودیت وزن دارایی‌ها را آزادتر کنید.\n"
                "- داده‌های قیمتی یا نوسان دارایی‌ها را بررسی کنید.\n"
            )
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
        st.markdown(f"---\n### {style} : {best_desc}")
        st.markdown(f"**{style_desc}**")
        if style in ["Sortino", "Omega"]:
            st.info(f"بازده هدف: {target_return*100:.2f}%")
        if style in ["CVaR", "VaR"]:
            st.info(f"سطح اطمینان: {int(cvar_alpha*100)}٪")
        # نمودار کارا فقط انگلیسی و حرفه‌ای
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
        # جدول وزن پرتفو
        st.markdown("#### جدول وزن پرتفو (سبد بهینه)")
        dfw = pd.DataFrame({'دارایی': asset_names, 'وزن (%)': np.round(best['weights']*100, 2)})
        st.dataframe(dfw.set_index('دارایی'), use_container_width=True)
        # خروجی‌ها
        st.markdown(f"""
        - **بازده سالانه:** {best['return']*100:.2f}%
        - **ریسک سالانه:** {best['risk']*100:.2f}%
        - **نسبت شارپ:** {best['sharpe']:.2f}
        - **سورتینو:** {best['sortino']:.2f}
        - **امگا:** {best['omega']:.2f}
        - **CVaR ({int(cvar_alpha*100)}%):** {best['cvar']*100:.2f}%
        - **VaR ({int(cvar_alpha*100)}%):** {best['var']*100:.2f}%
        - **Max Drawdown:** {best['drawdown']*100:.2f}%
        """)
        st.info(
            f"راهنمای استفاده از سبک {style}: {style_desc}\n"
            "در صورت مشاهده خطا، بازه ریسک، محدودیت وزن یا تعداد پرتفوها را تغییر دهید."
        )

    # پیش‌بینی قیمت انگلیسی و حرفه‌ای
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
        best_tuple = max(all_best, key=lambda x: (x[1]["return"], -x[1]["risk"]))
        style, best, desc = best_tuple
        st.success(f"سبد پیشنهادی: سبک '{style}' ({desc})")
        dfw = pd.DataFrame({'دارایی': asset_names, 'وزن (%)': np.round(best['weights']*100, 2)})
        st.dataframe(dfw.set_index('دارایی'), use_container_width=True)
        st.markdown(f"""
        - **بازده سالانه:** {best['return']*100:.2f}%
        - **ریسک سالانه:** {best['risk']*100:.2f}%
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
    - همه سبک‌های پرتفو همزمان محاسبه و مقایسه می‌شوند.<br>
    - برای هر سبک، مرز کارا فقط انگلیسی، خروجی و سبد بهینه و توضیحات فارسی نمایش داده می‌شود.<br>
    - پیش‌بینی قیمت برای هر دارایی به صورت حرفه‌ای و انگلیسی است.<br>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("لطفاً ابتدا داده قیمت دارایی‌ها را آپلود کنید.")
