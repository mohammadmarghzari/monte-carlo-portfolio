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
        name = file.name.split('.')[0]
        try:
            # خواندن فایل CSV
            df = pd.read_csv(file, thousands=',')
            df.columns = df.columns.str.strip().str.lower()

            # بررسی وجود ستون‌های مورد نیاز
            if 'date' not in df.columns or 'price' not in df.columns:
                st.error(f"فایل {name} باید شامل ستون‌های 'date' و 'price' باشد. ستون‌های موجود: {list(df.columns)}")
                continue

            # انتخاب ستون‌های تاریخ و قیمت
            df = df[['date', 'price']].copy()

            # تبدیل تاریخ به datetime و قیمت به عددی
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df.dropna(subset=['date', 'price'], inplace=True)

            if df.empty:
                st.error(f"فایل {name} پس از پردازش هیچ داده معتبری ندارد.")
                continue

            # تنظیم ایندکس
            df.set_index('date', inplace=True)
            prices_df[name] = df['price']
            asset_names.append(name)
        except Exception as e:
            st.error(f"خطا در پردازش فایل {name}: {e}")

    if prices_df.empty:
        st.error("❌ هیچ داده معتبری برای تحلیل وجود ندارد.")
        st.stop()

    # بررسی نوع ایندکس
    if not pd.api.types.is_datetime64_any_dtype(prices_df.index):
        st.error("⛔ ایندکس باید از نوع datetime باشد.")
        st.stop()

    st.subheader("🧾 پیش‌نمایش داده‌های قیمت")
    st.dataframe(prices_df.tail())

    returns = prices_df.pct_change().dropna()
    if returns.empty:
        st.error("❌ داده‌های کافی برای محاسبه بازده وجود ندارد.")
        st.stop()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # شبیه‌سازی مونت‌کارلو
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

    # انتخاب پرتفو با ریسک نزدیک ۳۰٪
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

    # نمودار سود و زیان
    st.subheader("📈 نمودار سود/زیان پورتفو نسبت به تغییر قیمت‌ها")

    price_changes = np.linspace(-0.5, 0.5, 100)
    total_change = np.zeros_like(price_changes)

    for i, w in enumerate(best_weights):
        total_change += w * price_changes

    plt.figure(figsize=(8, 4))
    plt.plot(price_changes * 100, total_change * 100, label="تغییر ارزش پرتفو")
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("درصد تغییر قیمت دارایی‌ها")
    plt.ylabel("درصد سود/زیان پرتفو")
    plt.title("نمودار سود/زیان پورتفو")
    plt.grid(True)
    st.pyplot(plt)

else:
    st.warning("لطفاً فایل‌های CSV با ستون‌های 'date' و 'price' آپلود کنید.")
