import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با روش مونت‌کارلو")
st.markdown("هدف: ساخت پرتفو با بازده بالا و ریسک کنترل‌شده")

def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.replace('%', '').str.lower()
        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)",
    type=['csv'],
    accept_multiple_files=True
)

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
if period == 'ماهانه':
    resample_rule = 'M'
    annual_factor = 12
elif period == 'سه‌ماهه':
    resample_rule = 'Q'
    annual_factor = 4
else:
    resample_rule = '2Q'
    annual_factor = 2

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []

    for file in uploaded_files:
        df = read_csv_file(file)
        if df is None:
            continue

        name = file.name.split('.')[0]

        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"فایل {name} باید دارای ستون‌های 'Date' و 'Price' باشد. ستون‌های یافت‌شده: {df.columns.tolist()}")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]

        if prices_df.empty:
            prices_df = df
        else:
            prices_df = prices_df.join(df, how='inner')

        asset_names.append(name)

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    st.subheader("🧪 پیش‌نمایش داده‌ها (آخرین قیمت‌های هر فایل)")
    st.dataframe(prices_df.tail())

    resampled_prices = prices_df.resample(resample_rule).last()
    resampled_prices = resampled_prices.apply(pd.to_numeric, errors='coerce')
    resampled_prices = resampled_prices.dropna()

    if resampled_prices.empty:
        st.error("❌ داده‌ها پس از بازنمونه‌گیری (resample) خالی شدند.")
        st.stop()

    returns = resampled_prices.pct_change().dropna()

    if returns.empty:
        st.error("❌ محاسبه بازده ممکن نیست. لطفاً فایل‌ها را بررسی کنید.")
        st.stop()

    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor

    # انحراف معیار سالانه برای هر دارایی
    asset_std_devs = np.sqrt(np.diag(cov_matrix))

    # 📌 تنظیم وزن ترجیحی دارایی‌ها بر اساس وضعیت بیمه
    use_put_option = st.checkbox("فعال‌سازی بیمه با آپشن پوت")
    if use_put_option:
        insurance_percent = st.number_input(
            "درصد پوشش بیمه (٪ از پرتفو)", min_value=0.0, max_value=100.0, value=30.0
        )
        effective_risk = asset_std_devs * (1 - insurance_percent / 100)
        preference_weights = asset_std_devs / effective_risk
    else:
        preference_weights = 1 / asset_std_devs

    preference_weights = preference_weights / np.sum(preference_weights)

    # شبیه‌سازی مونت‌کارلو
    np.random.seed(42)
    n_portfolios = 10000
    n_assets = len(asset_names)
    results = np.zeros((3 + n_assets, n_portfolios))

    for i in range(n_portfolios):
        random_factors = np.random.random(n_assets)
        weights = random_factors * preference_weights
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = port_return / port_std
        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3:, i] = weights

    target_risk = 0.30
    best_idx = np.argmin(np.abs(results[1] - target_risk))
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_weights = results[3:, best_idx]

    st.subheader("📊 نتایج پرتفو پیشنهادی (سالانه)")
    st.markdown(f"""
    - ✅ **بازده مورد انتظار سالانه:** {best_return:.2%}  
    - ⚠️ **ریسک سالانه (انحراف معیار):** {best_risk:.2%}  
    - 🧠 **نسبت شارپ:** {best_sharpe:.2f}
    """)
    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 **وزن {name}:** {best_weights[i]*100:.2f}%")

    st.subheader("📈 Portfolio Risk/Return Scatter Plot (Interactive)")
    fig = px.scatter(
        x=results[1]*100,
        y=results[0]*100,
        color=results[2],
        labels={'x': 'Annual Risk (%)', 'y': 'Annual Return (%)'},
        title='Simulated Portfolios',
        color_continuous_scale='Viridis'
    )
    fig.add_trace(go.Scatter(
        x=[best_risk * 100],
        y=[best_return * 100],
        mode='markers',
        marker=dict(color='red', size=12, symbol='star'),
        name='Optimal Portfolio'
    ))
    st.plotly_chart(fig)

else:
    st.warning("لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید.")
