import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با روش مونت‌کارلو")

st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader("چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    asset_names = [file.name.split('.')[0] for file in uploaded_files]

    st.sidebar.markdown("---")
    target_risk = st.sidebar.slider("🎯 ریسک هدف (درصد)", min_value=5, max_value=100, value=30, step=1) / 100
    n_portfolios = st.sidebar.number_input("📊 تعداد پرتفوهای شبیه‌سازی", min_value=1000, max_value=50000, value=10000, step=1000)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📏 محدودیت وزن دارایی‌ها")
    weight_limits = {}
    for name in asset_names:
        min_w, max_w = st.sidebar.slider(f"🔧 {name}", 0.0, 1.0, (0.0, 1.0), 0.05)
        weight_limits[name] = (min_w, max_w)

    prices_df = pd.DataFrame()
    for file, name in zip(uploaded_files, asset_names):
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()

        if 'Close' in df.columns:
            prices_df[name] = df['Close']
        else:
            st.error(f"❌ ستون 'Close' در فایل {name} پیدا نشد.")
            st.stop()

    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    n_assets = len(asset_names)

    np.random.seed(42)
    results = np.zeros((3 + n_assets, n_portfolios))
    count = 0
    while count < n_portfolios:
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        if all(weight_limits[name][0] <= w <= weight_limits[name][1] for w, name in zip(weights, asset_names)):
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = port_return / port_std
            results[0, count] = port_return
            results[1, count] = port_std
            results[2, count] = sharpe_ratio
            results[3:, count] = weights
            count += 1

    best_idx = np.argmin(np.abs(results[1] - target_risk))
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_weights = results[3:, best_idx]

    equal_weights = np.array([1/n_assets] * n_assets)
    equal_return = np.dot(equal_weights, mean_returns)
    equal_risk = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
    equal_sharpe = equal_return / equal_risk

    st.subheader("📊 مقایسه پرتفو پیشنهادی با پرتفو برابر")
    compare_df = pd.DataFrame({
        "": ["پرتفو پیشنهادی", "پرتفو برابر"],
        "بازده سالانه": [f"{best_return:.2%}", f"{equal_return:.2%}"],
        "ریسک سالانه": [f"{best_risk:.2%}", f"{equal_risk:.2%}"],
        "نسبت شارپ": [f"{best_sharpe:.2f}", f"{equal_sharpe:.2f}"]
    })
    st.table(compare_df)

    weights_df = pd.DataFrame({
        "دارایی": asset_names,
        "وزن پرتفو پیشنهادی (%)": best_weights * 100,
        "وزن پرتفو برابر (%)": equal_weights * 100
    })
    st.dataframe(weights_df)

    csv = weights_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 دانلود وزن پرتفو (CSV)", data=csv, file_name="portfolio_weights.csv", mime="text/csv")

    st.subheader("📈 مرز کارا و نقطه پرتفو بهینه")
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.4)
    ax.scatter(best_risk, best_return, color='red', label='پرتفو پیشنهادی')
    ax.scatter(equal_risk, equal_return, color='blue', label='پرتفو برابر')
    ax.set_xlabel("ریسک (انحراف معیار سالانه)")
    ax.set_ylabel("بازده سالانه")
    ax.set_title("مرز کارا")
    ax.grid(True)
    ax.legend()
    fig.colorbar(scatter, label="نسبت شارپ")
    st.pyplot(fig)

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
