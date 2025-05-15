import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="تحلیل پرتفو کامل", layout="wide")
st.title("📈 تحلیل پرتفو با بیمه آپشن، مونت‌کارلو، سود و زیان تخمینی و انحراف معیار")

st.sidebar.header("📂 فایل‌های CSV")
uploaded_files = st.sidebar.file_uploader("آپلود فایل‌های CSV (هر فایل یک دارایی)", type=["csv"], accept_multiple_files=True)

period = st.sidebar.selectbox("بازه تحلیل:", ['روزانه', 'ماهانه', 'سه‌ماهه'])
if period == 'روزانه':
    resample_rule, annual_factor = 'D', 252
elif period == 'ماهانه':
    resample_rule, annual_factor = 'M', 12
else:
    resample_rule, annual_factor = 'Q', 4

method = st.sidebar.radio("مدل تحلیل:", ["مونت‌کارلو", "مرز کارا (MPT)"])
use_option = st.sidebar.checkbox("📉 استفاده از بیمه با آپشن پوت")
target_risk = st.sidebar.slider("🎯 ریسک هدف (سالانه)", 5.0, 50.0, 25.0) / 100

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []

    for file in uploaded_files:
        name = file.name.split('.')[0]
        df = pd.read_csv(file, thousands=',')
        df.columns = df.columns.str.strip().str.lower()
        df.rename(columns={"date": "Date", "price": "Price"}, inplace=True)

        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.error(f"فایل {name} فاقد ستون‌های 'Date' و 'Price' است.")
            continue

        df = df[['Date', 'Price']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        df = df[['Price']]
        df.columns = [name]

        if prices_df.empty:
            prices_df = df
        else:
            prices_df = prices_df.join(df, how='inner')

        asset_names.append(name)

    if prices_df.empty:
        st.error("❌ هیچ داده معتبری برای تحلیل وجود ندارد.")
        st.stop()

    st.subheader("🧾 پیش‌نمایش داده‌های قیمت")
    st.dataframe(prices_df.tail())

    prices_resampled = prices_df.resample(resample_rule).last().dropna()
    returns = prices_resampled.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))

    option_data = {}
    if use_option:
        st.subheader("🛡 تنظیمات آپشن پوت")
        for asset in asset_names:
            st.markdown(f"#### {asset}")
            amount = st.number_input(f"🔹 مقدار دارایی پایه - {asset}", 0.0, 1e6, 1.0, 0.01, key=f"amt_{asset}")
            buy_price = st.number_input(f"💵 قیمت خرید دارایی - {asset}", 0.0, 1e6, 1000.0, 0.01, key=f"bp_{asset}")
            contracts = st.number_input(f"📄 تعداد قرارداد آپشن - {asset}", 0.0, 1e6, 0.0, 0.0001, key=f"opt_{asset}")
            strike = st.number_input(f"🎯 قیمت اعمال - {asset}", 0.0, 1e6, 1000.0, 0.01, key=f"strike_{asset}")
            premium = st.number_input(f"💰 قیمت هر آپشن - {asset}", 0.0, 1e6, 50.0, 0.01, key=f"premium_{asset}")
            base_val = amount * buy_price
            insured_val = contracts * strike
            coverage = min(insured_val / base_val, 1.0) if base_val > 0 else 0
            pnl = max(0, strike - buy_price) * contracts - contracts * premium
            option_data[asset] = {
                "amount": amount, "buy_price": buy_price,
                "contracts": contracts, "strike": strike,
                "premium": premium, "coverage": coverage,
                "pnl_ratio": pnl / base_val if base_val > 0 else 0
            }
        adj_returns = mean_returns + np.array([option_data[a]['pnl_ratio'] for a in asset_names])
        cov_adjustment = (1 - np.mean([option_data[a]['coverage'] for a in asset_names])) ** 2
        adj_cov = cov_matrix * cov_adjustment
    else:
        adj_returns = mean_returns
        adj_cov = cov_matrix

    n_assets = len(asset_names)
    n_portfolios = 5000
    results = np.zeros((3 + n_assets, n_portfolios))

    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        p_ret = np.dot(weights, adj_returns)
        p_risk = np.sqrt(np.dot(weights.T, np.dot(adj_cov, weights)))
        sharpe = p_ret / p_risk if p_risk > 0 else 0
        results[0, i] = p_ret
        results[1, i] = p_risk
        results[2, i] = sharpe
        results[3:, i] = weights

    idx = np.argmin(np.abs(results[1] - target_risk))
    best_ret, best_risk, best_sharpe = results[0, idx], results[1, idx], results[2, idx]
    best_weights = results[3:, idx]

    st.subheader("📊 نتایج پرتفو پیشنهادی")
    st.markdown(f"""
    ✅ بازده سالانه: {best_ret:.2%}  
    ⚠️ ریسک سالانه: {best_risk:.2%}  
    🧠 نسبت شارپ: {best_sharpe:.2f}
    """)
    for i, asset in enumerate(asset_names):
        st.markdown(f"🔹 وزن {asset}: {best_weights[i]*100:.2f}%")

    st.subheader("📈 نمودار مرز کارا / شبیه‌سازی")
    fig = px.scatter(
        x=results[1] * 100,
        y=results[0] * 100,
        color=results[2],
        labels={'x': 'Risk (%)', 'y': 'Expected Return (%)'},
        title="Efficient Frontier",
        color_continuous_scale='Viridis'
    )
    fig.add_trace(go.Scatter(
        x=[best_risk * 100],
        y=[best_ret * 100],
        mode='markers',
        marker=dict(color='red', size=12, symbol='star'),
        name='Target Portfolio'
    ))
    st.plotly_chart(fig)

    st.subheader("💵 سود و زیان دلاری تخمینی")
    capital = st.number_input("💰 مقدار سرمایه‌گذاری (دلار)", 0.0, 1e9, 10000.0, 100.0)
    profit = best_ret * capital
    loss = -best_risk * capital
    st.success(f"📈 سود تخمینی: {profit:,.2f} دلار")
    st.error(f"📉 زیان احتمالی در بدترین حالت (±1σ): {loss:,.2f} دلار")

    st.subheader("🎯 بازه بازده با احتمال 68% (±1 انحراف معیار)")
    min_return = best_ret - best_risk
    max_return = best_ret + best_risk
    st.info(f"درصدی: از {min_return:.2%} تا {max_return:.2%}")
    st.info(f"دلاری: از {capital * min_return:,.2f} تا {capital * max_return:,.2f}")

else:
    st.info("لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید.")
