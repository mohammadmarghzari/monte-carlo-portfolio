import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("📊 ابزار تحلیل پرتفو با بیمه اختیار فروش (Married Put)")

# --- بارگذاری داده‌ها ---
uploaded_files = st.file_uploader("📁 فایل‌های CSV را آپلود کنید", type="csv", accept_multiple_files=True)

prices_df = pd.DataFrame()
asset_names = []

if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        name = file.name.split('.')[0]
        asset_names.append(name)

        st.write(f"ستون‌های فایل {name}:", df.columns.tolist())

        if 'Date' not in df.columns:
            st.error(f"❌ ستون 'Date' در فایل {name} پیدا نشد.")
            st.stop()

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        df.set_index('Date', inplace=True)

        if 'Adj Close' in df.columns:
            prices_df[name] = df['Adj Close']
        elif 'Close' in df.columns:
            prices_df[name] = df['Close']
        else:
            st.error(f"❌ ستون 'Adj Close' یا 'Close' در فایل {name} پیدا نشد.")
            st.stop()

    st.success("✅ فایل‌ها با موفقیت بارگذاری و پردازش شدند.")

    returns = prices_df.pct_change().dropna()

    st.subheader("⚖️ تنظیم وزن دارایی‌ها")
    weights = []
    for asset in asset_names:
        weight = st.slider(f"وزن {asset}", 0.0, 1.0, 1.0 / len(asset_names), 0.01)
        weights.append(weight)
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    st.write(f"**📈 بازده سالانه پرتفو: {portfolio_return:.2%}**")
    st.write(f"**📉 ریسک سالانه پرتفو (انحراف معیار): {portfolio_std:.2%}**")

    st.subheader("🛡️ بیمه پرتفو با قرارداد اختیار فروش (Put Option)")
    use_option = st.checkbox("استفاده از بیمه (Married Put)", value=False)

    if use_option:
        insured_asset = st.selectbox("دارایی پایه برای بیمه:", asset_names)
        underlying_price = st.number_input("قیمت فعلی دارایی پایه:", min_value=0.0, value=100.0)
        strike_price = st.number_input("قیمت اعمال اختیار فروش:", min_value=0.0, value=90.0)
        premium = st.number_input("پرمیوم پرداختی برای هر قرارداد:", min_value=0.0, value=5.0)
        quantity = st.number_input("مقدار دارایی پایه تحت پوشش:", min_value=0.0, value=1.0)

        insured_index = asset_names.index(insured_asset)
        insured_weight = weights[insured_index]

        prices = np.linspace(underlying_price * 0.01, underlying_price * 20, 1000)
        base_values = prices * quantity
        put_payoffs = np.maximum(strike_price - prices, 0) * quantity - premium * quantity
        married_put_values = base_values + put_payoffs
        married_put_returns = (married_put_values - base_values[0]) / base_values[0] * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(prices, married_put_returns, label="سود/زیان Married Put", color="blue")
        ax.axhline(0, color="black", linestyle="--")
        ax.set_xlabel("قیمت دارایی پایه")
        ax.set_ylabel("درصد سود/زیان")
        ax.set_title("نمودار سود/زیان Married Put")
        ax.legend()
        st.pyplot(fig)

    st.success("✅ تحلیل پرتفو با موفقیت انجام شد.")
