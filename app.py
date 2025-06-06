import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="تحلیل پرتفو و بهینه‌سازی پیشرفته", layout="wide")
st.title("📊 ابزار جامع تحلیل پرتفو با بهینه‌سازی پیشرفته")

# ۱. انتخاب بازه زمانی دلخواه
st.sidebar.markdown("### ۱. بازه زمانی تحلیل")
date_range_mode = st.sidebar.radio("انتخاب بازه:", ["کل داده", "بازه دلخواه"])
date_start, date_end = None, None

if date_range_mode == "بازه دلخواه":
    date_start = st.sidebar.date_input("تاریخ شروع", value=datetime(2022,1,1))
    date_end = st.sidebar.date_input("تاریخ پایان", value=datetime.now())
    if date_end < date_start:
        st.sidebar.error("تاریخ پایان باید بعد از تاریخ شروع باشد.")

# ۲. بارگذاری و تنظیم دارایی‌ها
uploaded_files = st.sidebar.file_uploader("آپلود فایل‌های CSV با ستون‌های Date و Price", type="csv", accept_multiple_files=True)
asset_settings = {}

if uploaded_files:
    st.sidebar.markdown("### ۲. تنظیمات وزن و محدودیت هر دارایی")
    for file in uploaded_files:
        asset_name = file.name.split('.')[0]
        st.sidebar.markdown(f"**{asset_name}**")
        min_w = st.sidebar.slider(f"حداقل وزن {asset_name} (%)", 0.0, 100.0, 0.0, 1.0, key=f"min_{asset_name}")/100
        max_w = st.sidebar.slider(f"حداکثر وزن {asset_name} (%)", 0.0, 100.0, 100.0, 1.0, key=f"max_{asset_name}")/100
        init_w = st.sidebar.slider(f"وزن اولیه {asset_name} (%)", min_w*100, max_w*100, ((min_w+max_w)/2)*100, 1.0, key=f"init_{asset_name}")/100
        asset_settings[asset_name] = {"min": min_w, "max": max_w, "init": init_w}

st.sidebar.markdown("---")
st.sidebar.markdown("### ۳. تنظیمات بازه بازده")
period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]

# ۴. انتخاب روش بهینه‌سازی
st.sidebar.markdown("---")
st.sidebar.header("🎯 روش بهینه‌سازی پرتفو")
all_methods = [
    "واریانس میانگین (مرز کارا)",
    "کمینه CVaR (ارزش در معرض ریسک مشروط)",
    "برابری ریسک",
    "حداقل خطای ردیابی",
    "بیشینه نسبت اطلاعات",
    "بیشینه نرخ رشد هندسی (Kelly)",
    "بیشینه نسبت سورتینو",
    "بیشینه نسبت امگا",
    "کمترین افت سرمایه",
    "Black-Litterman"
]
method = st.sidebar.selectbox("روش بهینه‌سازی:", all_methods)
st.sidebar.markdown("---")

with st.sidebar.expander("⚙️ پارامترهای تخصصی هر روش"):
    if method in ["کمینه CVaR (ارزش در معرض ریسک مشروط)"]:
        cvar_alpha = st.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)
    else:
        cvar_alpha = 0.95
    if method in ["بیشینه نسبت سورتینو", "بیشینه نسبت امگا", "کمترین افت سرمایه"]:
        target_return = st.slider("حداقل بازده قابل قبول (%)", 0.0, 20.0, 5.0, 0.5)/100
    else:
        target_return = 0.0
    n_portfolios = st.slider("تعداد شبیه‌سازی پرتفوها", 1000, 10000, 3000, 1000)

# ۵. پردازش داده‌ها
prices_df = pd.DataFrame()
asset_names = []
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()
        if 'date' not in df.columns or 'price' not in df.columns:
            st.error(f"فایل {file.name} ستون‌های 'Date' و 'Price' ندارد.")
            continue
        df['Date'] = pd.to_datetime(df['date'])
        df['Price'] = pd.to_numeric(df['price'].astype(str).str.replace(",", ""), errors='coerce')
        df = df.dropna().set_index('Date')
        # فقط ستون Price را نگه دار تا تعداد ستون‌ها ۱ باشد
        df = df[['Price']]
        # حالا نام ستون را به نام دارایی تغییر بده
        name = file.name.split('.')[0]
        df.columns = [name]
        asset_names.append(name)
        if prices_df.empty:
            prices_df = df
        else:
            prices_df = prices_df.join(df, how='inner')
    prices_df = prices_df.dropna()

if not prices_df.empty:
    st.subheader("📈 قیمت‌های تعدیل‌شده دارایی‌ها")
    st.dataframe(prices_df.tail())
    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns_df = resampled_prices.pct_change().dropna()
    mean_returns = returns_df.mean() * annual_factor
    cov_matrix = returns_df.cov() * annual_factor

    st.subheader("📊 بازده دارایی‌ها")
    st.dataframe(returns_df.tail())

    tracking_index = returns_df.mean(axis=1).values

    results = []
    for w in np.random.dirichlet(np.ones(len(asset_names)), n_portfolios):
        legal = True
        for i, name in enumerate(asset_names):
            min_w, max_w = asset_settings[name]["min"], asset_settings[name]["max"]
            if not (min_w <= w[i] <= max_w):
                legal = False
        if not legal:
            continue
        port_ret = np.dot(w, mean_returns)
        port_risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sims = np.random.multivariate_normal(mean_returns / annual_factor, cov_matrix / annual_factor, 1000)
        sim_returns = np.dot(sims, w)
        var = np.percentile(sim_returns, (1 - cvar_alpha) * 100)
        cvar = sim_returns[sim_returns <= var].mean() if np.any(sim_returns <= var) else var
        downside = returns_df.copy()
        downside[downside > target_return] = 0
        downside_std = np.sqrt(np.dot(w.T, np.dot(downside.cov() * annual_factor, w)))
        sortino = (port_ret - target_return) / downside_std if downside_std > 0 else 0
        port_returns = np.dot(returns_df.values, w)
        omega = np.sum(np.maximum(port_returns - target_return, 0)) / (np.abs(np.sum(np.minimum(port_returns - target_return, 0))) + 1e-8)
        cum = (1 + port_returns).cumprod()
        peak = np.maximum.accumulate(cum)
        drawdowns = (cum - peak) / peak
        max_drawdown = drawdowns.min()
        excess = returns_df.sub(mean_returns.mean(), axis=1)
        info = (port_ret - mean_returns.mean()) / (np.std(np.dot(excess.values, w)) + 1e-8)
        kelly_growth = np.mean(np.log1p(port_returns))
        contrib = w * np.dot(cov_matrix, w)
        risk_budget = np.std(contrib) / (np.mean(contrib) + 1e-8)
        tracking_err = np.std(port_returns - tracking_index)
        results.append({
            "weights": w, "return": port_ret, "risk": port_risk, "cvar": cvar, "sortino": sortino,
            "omega": omega, "drawdown": max_drawdown, "info": info, "kelly": kelly_growth,
            "risk_budget": risk_budget, "tracking": tracking_err
        })

    if method == "واریانس میانگین (مرز کارا)":
        best = max(results, key=lambda x: x["return"] / x["risk"])
        explain = "پرتفو با بیشترین نسبت بازده به ریسک (Sharpe) روی مرز کارا."
    elif method == "کمینه CVaR (ارزش در معرض ریسک مشروط)":
        best = min(results, key=lambda x: x["cvar"])
        explain = "پرتفو با کمترین مقدار CVaR (خطر دنباله)."
    elif method == "برابری ریسک":
        best = min(results, key=lambda x: x["risk_budget"])
        explain = "پرتفو با نزدیک‌ترین سهم ریسک مساوی بین دارایی‌ها."
    elif method == "حداقل خطای ردیابی":
        best = min(results, key=lambda x: x["tracking"])
        explain = "پرتفو با کمترین خطای ردیابی نسبت به معیار پرتفو (میانگین دارایی‌ها)."
    elif method == "بیشینه نسبت اطلاعات":
        best = max(results, key=lambda x: x["info"])
        explain = "پرتفو با بیشترین نسبت اطلاعات نسبت به معیار پرتفو."
    elif method == "بیشینه نرخ رشد هندسی (Kelly)":
        best = max(results, key=lambda x: x["kelly"])
        explain = "پرتفو با بیشترین رشد هندسی مورد انتظار (معیار Kelly)."
    elif method == "بیشینه نسبت سورتینو":
        best = max(results, key=lambda x: x["sortino"])
        explain = "پرتفو با بیشترین نسبت سورتینو با حداقل بازده مقبول."
    elif method == "بیشینه نسبت امگا":
        best = max(results, key=lambda x: x["omega"])
        explain = "پرتفو با بیشترین نسبت امگا (فراتر از بازده مقبول)."
    elif method == "کمترین افت سرمایه":
        best = max(results, key=lambda x: x["drawdown"])
        explain = "پرتفو با کمترین افت سرمایه (حداقل Drawdown)."
    elif method == "Black-Litterman":
        best = max(results, key=lambda x: x["return"] / x["risk"])
        explain = "مدل Black-Litterman (در این نسخه: مرز کارا بدون دیدگاه خاص)."
    else:
        best = results[0]
        explain = "-"

    st.success(f"روش بهینه‌سازی: {method}")
    st.markdown(f"<div dir='rtl'>{explain}</div>", unsafe_allow_html=True)
    st.markdown(f"**📈 بازده پرتفو:** {best['return']:.2%}")
    st.markdown(f"**⚠️ ریسک پرتفو:** {best['risk']:.2%}")
    st.markdown(f"**📉 CVaR:** {best['cvar']:.2%}")
    st.markdown(f"**📊 نسبت سورتینو:** {best['sortino']:.2f}")
    st.markdown(f"**ℹ️ نسبت اطلاعات:** {best['info']:.2f}")
    st.markdown(f"**📈 رشد هندسی (Kelly):** {best['kelly']:.2%}")
    st.markdown(f"**📉 حداکثر افت سرمایه:** {best['drawdown']:.2%}")
    st.markdown(f"**⚖️ نسبت امگا:** {best['omega']:.2f}")
    st.markdown(f"**📊 تنوع ریسک:** {best['risk_budget']:.2f}")
    st.markdown(f"**🔁 خطای ردیابی:** {best['tracking']:.4f}")

    st.subheader("📌 نمودار توزیع وزن پرتفو بهینه")
    fig = go.Figure(data=[go.Pie(labels=asset_names, values=best['weights'], hole=0.5)])
    fig.update_layout(title="توزیع وزن دارایی‌ها")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌈 نمودار مرز کارا شبیه‌سازی‌شده")
    df = pd.DataFrame(results)
    fig2 = px.scatter(df, x="risk", y="return", color="sortino",
                      hover_data=["info", "omega", "cvar"], title="مرز کارا با رنگ‌بندی نسبت سورتینو")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📋 جدول وزن پرتفو بهینه")
    table = pd.DataFrame({"دارایی": asset_names, "وزن بهینه (%)": np.array(best['weights'])*100,
                          "حداقل وزن (%)": [asset_settings[n]["min"]*100 for n in asset_names],
                          "حداکثر وزن (%)": [asset_settings[n]["max"]*100 for n in asset_names]})
    st.dataframe(table.set_index("دارایی"), use_container_width=True, height=300)

else:
    st.info("برای شروع، فایل‌های داده با ستون‌های 'Date' و 'Price' را آپلود کنید.")
