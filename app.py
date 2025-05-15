import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با روش مونت‌کارلو")
st.markdown("ریسک هر دارایی = ۲۰٪ | هدف: ساخت پرتفو با ریسک نزدیک به ۳۰٪")

st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader("چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []

    for file in uploaded_files:
        df = pd.read_csv(file)
        name = file.name.split('.')[0]

        st.write(f"📄 فایل: {name} - ستون‌ها: {list(df.columns)}")

        # تشخیص ستون قیمت پایانی: 'close' یا 'price'
        possible_close_cols = [
            col for col in df.columns 
            if any(key in col.lower() for key in ['close', 'adj close', 'price'])
        ]

        if not possible_close_cols:
            st.error(f"❌ فایل '{name}' فاقد ستونی مشابه قیمت پایانی (مثل 'Price' یا 'Close') است.")
            st.stop()

        close_col = possible_close_cols[0]
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        df = df.dropna(subset=[close_col])

        st.success(f"✅ ستون انتخاب‌شده برای {name}: {close_col}")
        asset_names.append(name)
        prices_df[name] = df[close_col].reset_index(drop=True)

    # هماهنگ کردن طول داده‌ها
    min_len = min(len(col) for _, col in prices_df.items())
    prices_df = prices_df.iloc[:min_len]

    # محاسبه بازده روزانه و سالانه
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # مونت‌کارلو
    np.random.seed(42)
    n_portfolios = 10000
    n_assets = len(asset_names)
    results = np.zeros((3 + n_assets, n_portfolios))

    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = port_return / port_std

        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3:, i] = weights

    # انتخاب پرتفو با ریسک نزدیک به ۳۰٪
    target_risk = 0.30
    best_idx = np.argmin(np.abs(results[1] - target_risk))

    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_weights = results[3:, best_idx]

    st.subheader("📊 نتایج پرتفو پیشنهادی")
    st.markdown(f"""
    - ✅ **بازده مورد انتظار سالانه:** {best_return:.2%}  
    - ⚠️ **ریسک سالانه:** {best_risk:.2%}  
    - 🧠 **نسبت شارپ:** {best_sharpe:.2f}  
    """)

    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 **وزن {name}:** {best_weights[i]*100:.2f}٪")

    st.subheader("📈 نمودار سود/زیان پرتفو نسبت به تغییر قیمت‌ها")

    price_changes = np.linspace(-0.5, 0.5, 100)
    total_change = np.zeros_like(price_changes)

    for i, w in enumerate(best_weights):
        total_change += w * price_changes

    plt.figure(figsize=(8, 4))
    plt.plot(price_changes * 100, total_change * 100, label="تغییر ارزش پرتفو")
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("درصد تغییر قیمت دارایی‌ها")
    plt.ylabel("درصد سود/زیان پرتفو")
    plt.title("نمودار سود/زیان پرتفو")
    plt.grid(True)
    st.pyplot(plt)

else:
    st.warning("لطفاً چند فایل CSV قیمت دارایی‌ها آپلود کنید.")
