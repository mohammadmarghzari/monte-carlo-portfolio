
# نسخه نهایی ادغام‌شده: تحلیل پرتفو + ویژگی‌های پیشرفته بهینه‌سازی

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import json
import base64

st.set_page_config(page_title="تحلیل پرتفو و بهینه‌سازی پیشرفته", layout="wide")
st.title("📊 ابزار جامع تحلیل پرتفو با بهینه‌سازی پیشرفته")

st.sidebar.header("📥 بارگذاری داده‌ها")

uploaded_files = st.sidebar.file_uploader("آپلود فایل‌های CSV با ستون‌های Date و Price", type="csv", accept_multiple_files=True)
period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]

returns_df = pd.DataFrame()
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
        name = file.name.split('.')[0]
        df.columns = [name]
        asset_names.append(name)
        if prices_df.empty:
            prices_df = df
        else:
            prices_df = prices_df.join(df, how='inner')

    prices_df = prices_df.dropna()
    st.subheader("📈 قیمت‌های تعدیل‌شده دارایی‌ها")
    st.dataframe(prices_df.tail())

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns_df = resampled_prices.pct_change().dropna()
    mean_returns = returns_df.mean() * annual_factor
    cov_matrix = returns_df.cov() * annual_factor

    st.subheader("📊 بازده دارایی‌ها")
    st.dataframe(returns_df.tail())

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ تنظیمات بهینه‌سازی")
    method = st.sidebar.selectbox("روش بهینه‌سازی پرتفو", [
        "واریانس میانگین (مرز کارا)",
        "کمینه CVaR (ارزش در معرض ریسک مشروط)",
        "برابری ریسک",
        "حداقل خطای ردیابی",
        "بیشینه نسبت اطلاعات",
        "بیشینه نرخ رشد هندسی (Kelly)",
        "بیشینه نسبت سورتینو",
        "بیشینه نسبت امگا",
        "کمترین افت سرمایه"
    ])
    target_return = st.sidebar.slider("حداقل بازده قابل قبول", 0.0, 0.2, 0.05, 0.01)
    cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.8, 0.99, 0.95, 0.01)
    n_portfolios = 3000

    def evaluate(weights):
        ret = np.dot(weights, mean_returns)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        downside = returns_df.copy()
        downside[downside > target_return] = 0
        downside_std = np.sqrt(np.dot(weights.T, np.dot(downside.cov() * annual_factor, weights)))
        sortino = (ret - target_return) / downside_std if downside_std > 0 else 0
        excess = returns_df.sub(mean_returns.mean(), axis=1)
        info = (ret - mean_returns.mean()) / np.std(np.dot(excess.values, weights))
        growth = np.mean(np.log1p(np.dot(returns_df.values, weights)))
        min_draw = np.min(np.dot(returns_df.values, weights))
        omega = np.sum(np.maximum(np.dot(returns_df.values, weights) - target_return, 0)) /                 np.abs(np.sum(np.minimum(np.dot(returns_df.values, weights) - target_return, 0)))
        sims = np.random.multivariate_normal(mean_returns / annual_factor, cov_matrix / annual_factor, 1000)
        sim_returns = np.dot(sims, weights)
        var = np.percentile(sim_returns, (1 - cvar_alpha) * 100)
        cvar = sim_returns[sim_returns <= var].mean()
        contrib = weights * np.dot(cov_matrix, weights)
        budget_std = np.std(contrib) / np.mean(contrib)
        track_err = np.std(np.dot(returns_df.values, weights) - returns_df.mean(axis=1).values)
        return {
            "weights": weights, "return": ret, "risk": std,
            "sortino": sortino, "info": info, "growth": growth,
            "drawdown": min_draw, "omega": omega, "cvar": cvar,
            "risk_budget": budget_std, "tracking": track_err
        }

    results = [evaluate(w) for w in np.random.dirichlet(np.ones(len(asset_names)), n_portfolios)]

    if method == "واریانس میانگین (مرز کارا)":
        best = max(results, key=lambda x: x["return"] / x["risk"])
    elif method == "کمینه CVaR (ارزش در معرض ریسک مشروط)":
        best = min(results, key=lambda x: x["cvar"])
    elif method == "برابری ریسک":
        best = min(results, key=lambda x: x["risk_budget"])
    elif method == "حداقل خطای ردیابی":
        best = min(results, key=lambda x: x["tracking"])
    elif method == "بیشینه نسبت اطلاعات":
        best = max(results, key=lambda x: x["info"])
    elif method == "بیشینه نرخ رشد هندسی (Kelly)":
        best = max(results, key=lambda x: x["growth"])
    elif method == "بیشینه نسبت سورتینو":
        best = max(results, key=lambda x: x["sortino"])
    elif method == "بیشینه نسبت امگا":
        best = max(results, key=lambda x: x["omega"])
    elif method == "کمترین افت سرمایه":
        best = max(results, key=lambda x: x["drawdown"])

    st.subheader("🔍 خلاصه پرتفو بهینه")
    st.markdown(f"📈 بازده: {best['return']:.2%}")
    st.markdown(f"⚠️ ریسک: {best['risk']:.2%}")
    st.markdown(f"📉 CVaR: {best['cvar']:.2%}")
    st.markdown(f"📊 نسبت سورتینو: {best['sortino']:.2f}")
    st.markdown(f"ℹ️ نسبت اطلاعات: {best['info']:.2f}")
    st.markdown(f"📈 رشد هندسی: {best['growth']:.2%}")
    st.markdown(f"📉 حداقل بازده (افت): {best['drawdown']:.2%}")
    st.markdown(f"⚖️ نسبت امگا: {best['omega']:.2f}")
    st.markdown(f"📊 تنوع ریسک: {best['risk_budget']:.2f}")
    st.markdown(f"🔁 خطای ردیابی: {best['tracking']:.4f}")

    st.subheader("📌 نمودار توزیع وزن پرتفو بهینه")
    fig = go.Figure(data=[go.Pie(labels=asset_names, values=best['weights'], hole=0.5)])
    fig.update_layout(title="توزیع وزن دارایی‌ها")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🌈 نمودار مرز کارا شبیه‌سازی‌شده")
    df = pd.DataFrame(results)
    fig2 = px.scatter(df, x="risk", y="return", color="sortino",
                      hover_data=["info", "omega", "cvar"], title="مرز کارا با رنگ‌بندی نسبت سورتینو")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("برای شروع، فایل‌های داده با ستون‌های 'Date' و 'Price' را آپلود کنید.")
