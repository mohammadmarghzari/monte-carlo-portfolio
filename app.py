import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با روش مونت‌کارلو")
st.markdown("ریسک هر دارایی = ۲۰٪ | هدف: ساخت پرتفو با ریسک نزدیک به ۳۰٪")

st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
st.sidebar.markdown("""
### 📋 راهنمای آپلود فایل‌ها
- فایل‌های CSV باید شامل ستونی برای قیمت (مثل 'Close' یا 'Price') باشند.
- داده‌ها باید عددی و بدون مقادیر گمشده باشند.
- برای بهترین نتیجه، از داده‌های هم‌زمان استفاده کنید.
""")
uploaded_files = st.sidebar.file_uploader("چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True)

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []

    for file in uploaded_files:
        df = pd.read_csv(file)
        name = file.name.split('.')[0]

        if df.empty:
            st.error(f"❌ فایل '{name}' خالی است.")
            st.stop()

        st.write(f"📄 فایل: {name} - ستون‌ها: {list(df.columns)}")

        possible_close_cols = [
            col for col in df.columns 
            if any(key in col.lower() for key in ['close', 'adj close', 'price'])
        ]

        if not possible_close_cols:
            st.error(f"❌ فایل '{name}' فاقد ستونی مشابه قیمت پایانی (مثل 'Price' یا 'Close') است.")
            st.stop()

        close_col = possible_close_cols[0]
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')

        if df[close_col].isna().all():
            st.error(f"❌ ستون '{close_col}' در فایل '{name}' فاقد داده‌های معتبر است.")
            st.stop()

        df = df.dropna(subset=[close_col])
        st.success(f"✅ ستون انتخاب‌شده برای {name}: {close_col}")
        asset_names.append(name)
        prices_df[name] = df[close_col].reset_index(drop=True)

    # هماهنگ کردن طول داده‌ها
    min_len = min(len(col) for _, col in prices_df.items())
    if min_len < 100:
        st.warning("⚠️ تعداد داده‌های معتبر بسیار کم است. تحلیل ممکن است نادرست باشد.")
    prices_df = prices_df.iloc[:min_len]

    # بررسی داده‌های منفی یا صفر
    if (prices_df <= 0).any().any():
        st.error("❌ برخی قیمت‌ها منفی یا صفر هستند. لطفاً داده‌ها را بررسی کنید.")
        st.stop()

    # محاسبه بازده روزانه و سالانه
    returns = prices_df.pct_change().dropna()
    if returns.empty:
        st.error("❌ داده‌های کافی برای محاسبه بازده وجود ندارد.")
        st.stop()

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # مونت‌کارلو
    np.random.seed(42)
    n_portfolios = st.sidebar.slider("تعداد شبیه‌سازی‌های مونت‌کارلو", 1000, 50000, 10000)
    n_assets = len(asset_names)
    
    if n_assets < 2:
        st.error("❌ حداقل دو دارایی برای تحلیل پرتفو موردنیاز است.")
        st.stop()

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

    if np.any(np.isnan([best_return, best_risk, best_sharpe])) or np.any(np.isinf([best_return, best_risk, best_sharpe])):
        st.error("❌ مقادیر محاسباتی نامعتبر هستند. لطفاً داده‌ها را بررسی کنید.")
        st.stop()

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

    try:
        plt.figure(figsize=(8, 4))
        plt.plot(price_changes * 100, total_change * 100, label="تغییر ارزش پرتفو")
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel("درصد تغییر قیمت دارایی‌ها")
        plt.ylabel("درصد سود/زیان پرتفو")
        plt.title("نمودار سود/زیان پرتفو")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)
    except Exception as e:
        st.error(f"❌ خطا در رسم نمودار: {str(e)}")

else:
    st.warning("لطفاً چند فایل CSV قیمت دارایی‌ها آپلود کنید.")
