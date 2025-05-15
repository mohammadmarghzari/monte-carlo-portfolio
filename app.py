import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="تحلیل سبد رمزارز", layout="wide")
st.title("🚀 ابزار تحلیل سبد رمزارز با شبیه‌سازی مونت‌کارلو")

st.sidebar.header("📥 بارگذاری فایل‌های قیمت رمزارز (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "لطفاً فایل‌های CSV شامل ستون‌های تاریخ و قیمت را آپلود کنید",
    type=["csv"],
    accept_multiple_files=True
)

period = st.sidebar.selectbox("بازه تحلیل:", ['روزانه', 'هفتگی', 'ماهانه'])
resample_dict = {'روزانه': ('D', 365), 'هفتگی': ('W', 52), 'ماهانه': ('M', 12)}
resample_rule, annual_factor = resample_dict[period]

if uploaded_files:
    prices_df = pd.DataFrame()
    coins = []

    for file in uploaded_files:
        coin_name = file.name.split('.')[0]
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()
        if 'date' not in df.columns or 'price' not in df.columns:
            st.error(f"فایل {coin_name} باید شامل ستون‌های 'date' و 'price' باشد.")
            continue
        df = df[['date', 'price']].copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df.dropna(subset=['date', 'price'], inplace=True)
        if df.empty:
            st.error(f"فایل {coin_name} داده معتبر ندارد.")
            continue
        df.set_index('date', inplace=True)
        df.rename(columns={'price': coin_name}, inplace=True)
        if prices_df.empty:
            prices_df = df
        else:
            prices_df = prices_df.join(df, how='inner')
        coins.append(coin_name)

    if prices_df.empty:
        st.error("هیچ داده‌ای برای تحلیل وجود ندارد.")
        st.stop()

    st.subheader("پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail())

    prices_resampled = prices_df.resample(resample_rule).last().dropna()
    returns = prices_resampled.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor

    n_assets = len(coins)
    n_portfolios = 3000
    results = np.zeros((3 + n_assets, n_portfolios))
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        ret = np.dot(weights, mean_returns)
        risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / risk if risk != 0 else 0
        results[0, i] = ret
        results[1, i] = risk
        results[2, i] = sharpe
        results[3:, i] = weights

    idx = np.argmax(results[2])
    best_ret, best_risk, best_sharpe = results[0, idx], results[1, idx], results[2, idx]
    best_weights = results[3:, idx]

    st.subheader("📊 بهترین سبد پیشنهادی")
    st.markdown(f"بازده سالانه: {best_ret:.2%}")
    st.markdown(f"ریسک سالانه: {best_risk:.2%}")
    st.markdown(f"نسبت شارپ: {best_sharpe:.2f}")
    for i, coin in enumerate(coins):
        st.markdown(f"وزن {coin}: {best_weights[i]*100:.2f}%")

    fig = px.scatter(
        x=results[1]*100,
        y=results[0]*100,
        color=results[2],
        labels={"x": "ریسک (%)", "y": "بازده مورد انتظار (%)"},
        title="مرز کارای سبد رمزارزها",
        color_continuous_scale="Viridis"
    )
    fig.add_scatter(
        x=[best_risk*100], y=[best_ret*100],
        marker=dict(color="red", size=12, symbol="star"),
        mode="markers", name="سبد بهینه"
    )
    st.plotly_chart(fig)

    st.subheader("💵 سود و زیان تخمینی (دلاری)")
    capital = st.number_input("مقدار کل سرمایه (دلار):", 0.0, 1e9, 10000.0, 100.0)
    st.success(f"سود تخمینی سالانه: {best_ret*capital:,.2f} دلار")
    st.error(f"زیان احتمالی (±1σ): {-best_risk*capital:,.2f} دلار")
else:
    st.info("لطفاً فایل‌های قیمت رمزارز را بارگذاری کنید.")
