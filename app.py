import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="تحلیل پرتفو پیشرفته", layout="wide")
st.title("📊 ابزار پیشرفته تحلیل پرتفو و مدیریت ریسک")

# ----- تنظیمات سایدبار -----
st.sidebar.header("🔧 تنظیمات تحلیل")

# انتخاب بازه زمانی دلخواه
date_start = st.sidebar.date_input("تاریخ شروع تحلیل", value=datetime(2022, 1, 1))
date_end = st.sidebar.date_input("تاریخ پایان تحلیل", value=datetime.today())
if date_end < date_start:
    st.sidebar.error("تاریخ پایان باید بعد از تاریخ شروع باشد.")

# انتخاب روش بهینه‌سازی
st.sidebar.markdown("---")
all_methods = [
    "Sharpe (مرز کارا)", "Sortino", "Omega", "CVaR", "VaR", "Max Drawdown"
]
method = st.sidebar.selectbox("روش بهینه‌سازی:", all_methods)

# پارامترهای تخصصی هر روش
st.sidebar.markdown("### ⚙️ پارامترهای روش انتخابی")
target_return = st.sidebar.number_input("حداقل بازده مقبول (%)", value=5.0, min_value=0.0, max_value=100.0, step=0.1) / 100
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR/VaR", 0.80, 0.99, 0.95, 0.01)
n_portfolios = st.sidebar.slider("تعداد پرتفوهای شبیه‌سازی", 1000, 20000, 5000, 1000)

# بارگذاری داده‌ها (CSV یا یاهو فاینانس)
st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader("آپلود فایل CSV (ستون Date و Price)", type=['csv'], accept_multiple_files=True)
with st.sidebar.expander("دریافت داده از یاهو فاینانس"):
    tickers = st.text_input("نمادها (مثال: BTC-USD,AAPL,ETH-USD)", "")
    if st.button("دانلود داده"):
        data_yf = yf.download(
            tickers=[t.strip() for t in tickers.split(",") if t.strip()],
            start=str(date_start), end=str(date_end)
        )
        if not data_yf.empty:
            for t in tickers.split(","):
                t = t.strip()
                if t == "":
                    continue
                if (t,) in data_yf.columns:  # For single asset, data is not multi-indexed
                    df = data_yf.reset_index()[['Date', 'Adj Close']].rename(columns={'Adj Close': 'Price'})
                else:
                    df = data_yf[t].reset_index()[['Date', 'Adj Close']].rename(columns={'Adj Close': 'Price'})
                df.to_csv(f"{t}.csv", index=False)
                st.success(f"داده {t} با موفقیت آماده شد. می‌توانید آن را به ابزار آپلود کنید.")

# پردازش داده‌ها
prices_df = pd.DataFrame()
asset_names = []
weight_settings = {}

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()
        if 'date' not in df.columns or 'price' not in df.columns:
            st.error(f"فایل {file.name} باید 'Date' و 'Price' داشته باشد.")
            continue
        df['Date'] = pd.to_datetime(df['date'])
        df = df[(df['Date'] >= pd.to_datetime(date_start)) & (df['Date'] <= pd.to_datetime(date_end))]
        name = file.name.split('.')[0]
        df = df[['Date', 'price']].rename(columns={'price': name}).set_index('Date')
        asset_names.append(name)
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
    prices_df.dropna(inplace=True)

    # محدودیت و وزن اولیه برای هر دارایی
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚖️ محدودیت و وزن اولیه هر دارایی")
    for name in asset_names:
        min_w = st.sidebar.number_input(f"حداقل وزن {name} (%)", 0.0, 100.0, 0.0, 1.0, key=f"min_{name}") / 100
        max_w = st.sidebar.number_input(f"حداکثر وزن {name} (%)", 0.0, 100.0, 100.0, 1.0, key=f"max_{name}") / 100
        init_w = st.sidebar.number_input(f"وزن اولیه {name} (%)", min_w*100, max_w*100, ((min_w+max_w)/2)*100, 1.0, key=f"init_{name}") / 100
        weight_settings[name] = {'min': min_w, 'max': max_w, 'init': init_w}

# تحلیل پرتفو
if not prices_df.empty:
    st.markdown("### 📈 داده‌های قیمت دارایی‌ها")
    st.dataframe(prices_df.tail())

    returns = prices_df.pct_change().dropna()
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252

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
        port_ret = np.dot(w, mean_ret)
        port_risk = np.sqrt(np.dot(w.T, np.dot(cov, w)))
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
        results.append({
            "weights": w, "return": port_ret, "risk": port_risk, "sortino": sortino,
            "omega": omega, "cvar": cvar, "var": var, "drawdown": max_dd
        })

    # انتخاب پرتفو بهینه بر اساس روش انتخابی
    if method == "Sharpe (مرز کارا)":
        best = max(results, key=lambda x: x["return"]/x["risk"])
        explain = "پرتفو با بیشترین نسبت شارپ (مرز کارا)."
    elif method == "Sortino":
        best = max(results, key=lambda x: x["sortino"])
        explain = "پرتفو با بیشترین نسبت سورتینو (توجه به زیان پایین‌تر از بازده هدف)."
    elif method == "Omega":
        best = max(results, key=lambda x: x["omega"])
        explain = "پرتفو با بیشترین نسبت امگا (سود به زیان)."
    elif method == "CVaR":
        best = max(results, key=lambda x: x["cvar"])
        explain = "پرتفو با کمترین CVaR (ارزش در معرض ریسک مشروط)."
    elif method == "VaR":
        best = min(results, key=lambda x: x["var"])
        explain = "پرتفو با کمترین VaR (ارزش در معرض ریسک)."
    elif method == "Max Drawdown":
        best = max(results, key=lambda x: x["drawdown"])
        explain = "پرتفو با کمترین افت سرمایه (Max Drawdown)."
    else:
        best = results[0]
        explain = "-"

    # نمایش نتایج
    st.markdown(f"### 📊 خلاصه پرتفو بهینه ({method})")
    st.markdown(f"- **بازده سالانه:** {best['return']:.2%}")
    st.markdown(f"- **ریسک سالانه:** {best['risk']:.2%}")
    st.markdown(f"- **سورتینو:** {best['sortino']:.2f}")
    st.markdown(f"- **امگا:** {best['omega']:.2f}")
    st.markdown(f"- **CVaR ({int(cvar_alpha*100)}%):** {best['cvar']:.2%}")
    st.markdown(f"- **VaR ({int(cvar_alpha*100)}%):** {best['var']:.2%}")
    st.markdown(f"- **Max Drawdown:** {best['drawdown']:.2%}")
    st.markdown(f"**{explain}**")

    st.markdown("#### 📋 جدول وزن پرتفو")
    dfw = pd.DataFrame({'دارایی': asset_names, 'وزن (%)': np.round(best['weights']*100, 2)})
    st.dataframe(dfw.set_index('دارایی'), use_container_width=True)

    # نمودارها
    st.markdown("#### 📈 نمودار وزنی پرتفو")
    fig = go.Figure(data=[go.Pie(labels=asset_names, values=best['weights'], hole=0.5)])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🌈 مرز کارا و Max Drawdown")
    df_res = pd.DataFrame(results)
    fig2 = px.scatter(df_res, x="risk", y="return", color="drawdown", labels={'drawdown': 'Max Drawdown'})
    fig2.add_trace(go.Scatter(x=[best["risk"]], y=[best["return"]], mode="markers+text", marker=dict(size=14, color="red"), text=["پرتفو بهینه"], textposition="top center"))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div dir="rtl" style="text-align:right;">
    <b>توضیحات:</b><br>
    - همه پارامترهای مهم هر روش از جمله بازده مقبول (target_return)، سطح اطمینان CVaR/VaR، تعداد پرتفو و محدودیت وزن هر دارایی قابل تنظیم است.<br>
    - بازه زمانی تحلیل با انتخاب تاریخ شروع و پایان تعیین می‌شود.<br>
    - وزن اولیه و محدودیت برای هر دارایی جداگانه در سایدبار وجود دارد.<br>
    - روش‌های Max Drawdown و VaR و سایر روش‌ها به عنوان روش بهینه‌سازی و همچنین مقدار خروجی نمایش داده می‌شوند.<br>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("لطفاً داده بارگذاری کنید یا از یاهو فاینانس دانلود نمایید.")
