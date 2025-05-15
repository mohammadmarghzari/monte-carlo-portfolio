import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# تنظیمات صفحه
st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با روش مونت‌کارلو")
st.markdown("هدف: ساخت پرتفو با بازده بالا و ریسک کنترل‌شده")

# تابع خواندن فایل CSV
def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

# سایدبار برای آپلود فایل‌ها
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)",
    type=['csv'],
    accept_multiple_files=True
)

# انتخاب بازه زمانی تحلیل
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

# اگر فایل‌ها آپلود شدند
if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []

    for file in uploaded_files:
        df = read_csv_file(file)
        if df is None:
            continue

        name = file.name.split('.')[0]
        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"فایل {name} باید دارای ستون‌های 'Date' و 'Price' باشد.")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
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

    # محاسبه بازده‌ها
    resampled_prices = prices_df.resample(resample_rule).last()
    resampled_prices = resampled_prices.apply(pd.to_numeric, errors='coerce')
    resampled_prices = resampled_prices.dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor

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

    # انتخاب پرتفو با ریسک هدف
    target_risk = 0.30
    best_idx = np.argmin(np.abs(results[1] - target_risk))
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_weights = results[3:, best_idx]

    # مدیریت ریسک با درصد پوشش بیمه‌ای آپشن پوت
    st.subheader("🛡 مدیریت ریسک با آپشن پوت")
    use_put_option = st.checkbox("فعال‌سازی بیمه با آپشن پوت")

    if use_put_option:
        insurance_percent = st.number_input(
            "درصد پوشش بیمه (درصدی از پرتفو که بیمه می‌شود)", 
            min_value=0.0, max_value=100.0, value=30.0, step=1.0
        )
        adjusted_risk = best_risk * (1 - (insurance_percent / 100))
        adjusted_sharpe = best_return / adjusted_risk if adjusted_risk != 0 else 0
        risk_for_display = adjusted_risk
        sharpe_for_display = adjusted_sharpe
        st.info(f"✅ ریسک بیمه‌شده: {adjusted_risk:.2%} | نسبت شارپ جدید: {adjusted_sharpe:.2f}")
    else:
        risk_for_display = best_risk
        sharpe_for_display = best_sharpe

    # نمایش نتایج پرتفو
    st.subheader("📊 نتایج پرتفو پیشنهادی (سالانه)")
    st.markdown(f"""
    - ✅ **بازده مورد انتظار سالانه:** {best_return:.2%}  
    - ⚠️ **ریسک سالانه (انحراف معیار):** {risk_for_display:.2%}  
    - 🧠 **نسبت شارپ:** {sharpe_for_display:.2f}
    """)
    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 **وزن {name}:** {best_weights[i]*100:.2f}%")

    # نمودار پراکندگی تعاملی به انگلیسی
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
        x=[risk_for_display * 100],
        y=[best_return * 100],
        mode='markers',
        marker=dict(color='red', size=12, symbol='star'),
        name='Optimal Portfolio'
    ))
    st.plotly_chart(fig)

else:
    st.warning("لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید.")
