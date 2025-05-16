# ابزار تحلیل پرتفو با بیمه Married Put و بهینه‌سازی کامل

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm

st.set_page_config(page_title="تحلیل پرتفو با Married Put", layout="wide")
st.title("🛡️ تحلیل پرتفو با بیمه Married Put و شبیه‌سازی مونت‌کارلو")

st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)",
    type=['csv'],
    accept_multiple_files=True
)

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]

def read_csv(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
    df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Price'] = df['Price'].astype(str).str.replace(',', '')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['Date', 'Price'], inplace=True)
    return df.set_index('Date')

if uploaded_files:
    price_df = pd.DataFrame()
    assets = []
    insurance_data = {}

    for file in uploaded_files:
        name = file.name.split('.')[0]
        df = read_csv(file)
        df.columns = [name]
        if price_df.empty:
            price_df = df
        else:
            price_df = price_df.join(df, how='inner')
        assets.append(name)

        with st.expander(f"⚙️ بیمه برای {name}"):
            use_insurance = st.checkbox(f"فعال‌سازی بیمه برای {name}", key=f"ins_{name}")
            if use_insurance:
                premium_pct = st.number_input("درصد پرمیوم از قیمت پایه", 0.0, 1.0, 0.02, step=0.001, format="%.3f", key=f"prem_{name}")
                strike_price = st.number_input("قیمت اعمال پوت", 0.0, None, value=0.0, key=f"strike_{name}")
                base_price = st.number_input("قیمت فعلی دارایی پایه", 0.0, None, value=df.iloc[-1, 0], key=f"base_{name}")
                base_qty = st.number_input("مقدار دارایی پایه", 0.0, None, value=1.0, step=0.01, format="%.2f", key=f"qty_{name}")
                insurance_data[name] = {
                    'premium_pct': premium_pct,
                    'strike': strike_price,
                    'base_price': base_price,
                    'qty': base_qty
                }

    price_df = price_df.resample(resample_rule).last().dropna()
    returns = price_df.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor

    def calculate_married_put_return(base_price, strike, premium_pct, qty):
        premium = base_price * premium_pct
        expected_drop = min(base_price - strike, 0)
        payoff = max(strike - base_price, 0)
        total_pnl = (expected_drop + payoff - premium) * qty
        return total_pnl / (base_price * qty)

    adj_returns = mean_returns.copy()
    for name, info in insurance_data.items():
        insurance_effect = calculate_married_put_return(
            info['base_price'], info['strike'], info['premium_pct'], info['qty']
        )
        adj_returns[name] += insurance_effect

    st.subheader("📈 میانگین بازده سالانه (پس از بیمه)")
    st.dataframe(adj_returns.apply(lambda x: f"{x:.2%}"))

    n_assets = len(assets)
    n_portfolios = 10000
    results = np.zeros((3 + n_assets, n_portfolios))
    np.random.seed(42)

    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        port_return = np.dot(weights, adj_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_std
        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe
        results[3:, i] = weights

    best_idx = np.argmax(results[2])
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_weights = results[3:, best_idx]

    st.subheader("📊 پرتفو بهینه (بیشترین شارپ)")
    st.write(f"بازده: {best_return:.2%}")
    st.write(f"ریسک: {best_risk:.2%}")
    for i, name in enumerate(assets):
        st.write(f"🔹 {name}: {best_weights[i]*100:.2f}%")

    fig = px.scatter(
        x=results[1]*100,
        y=results[0]*100,
        color=results[2],
        labels={'x': 'ریسک سالانه (%)', 'y': 'بازده سالانه (%)'},
        title='مرز کارا و شبیه‌سازی مونت‌کارلو',
        color_continuous_scale='Viridis'
    )
    fig.add_trace(go.Scatter(
        x=[best_risk*100],
        y=[best_return*100],
        mode='markers',
        marker=dict(color='red', size=12, symbol='star'),
        name='پرتفو بهینه'
    ))
    st.plotly_chart(fig)

    st.subheader("📸 ذخیره نمودار")
    save = st.button("📥 ذخیره نمودار به عنوان تصویر")
    if save:
        import io
        from PIL import Image
        img_bytes = fig.to_image(format="png")
        st.download_button(
            label="📷 دانلود تصویر",
            data=img_bytes,
            file_name="efficient_frontier.png",
            mime="image/png"
        )
else:
    st.warning("لطفاً فایل‌های CSV با ستون‌های Date و Price را بارگذاری کنید.")
