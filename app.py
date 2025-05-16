import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با روش مونت‌کارلو")

# تابع خواندن و پاکسازی فایل CSV
def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        # پاکسازی ستون‌ها
        df.columns = df.columns.str.strip().str.lower()
        # پیدا کردن ستون‌های تاریخ و قیمت (غیر حساس به حروف)
        date_col = next((c for c in df.columns if 'date' in c), None)
        price_col = next((c for c in df.columns if 'price' in c), None)
        if date_col is None or price_col is None:
            st.error(f"فایل {file.name} ستون‌های Date و Price لازم را ندارد.")
            return None
        df = df[[date_col, price_col]]
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
        df = df.dropna(subset=['Price'])
        df = df.set_index('Date').sort_index()
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

st.sidebar.header("📂 آپلود فایل‌های دارایی (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "یک یا چند فایل CSV آپلود کنید (هر فایل یک دارایی)", 
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
    # خواندن فایل‌ها و ادغام داده‌ها
    prices_df = pd.DataFrame()
    asset_names = []
    for file in uploaded_files:
        df = read_csv_file(file)
        if df is not None:
            name = file.name.split('.')[0]
            df.columns = [name]
            if prices_df.empty:
                prices_df = df
            else:
                prices_df = prices_df.join(df, how='inner')
            asset_names.append(name)
    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    st.subheader("📊 داده‌های آخرین قیمت‌های هر دارایی")
    st.dataframe(prices_df.tail())

    # بازنمونه‌گیری (resample)
    prices_resampled = prices_df.resample(resample_rule).last().dropna()
    returns = prices_resampled.pct_change().dropna()

    if returns.empty:
        st.error("❌ محاسبه بازده ممکن نیست، داده‌ها کافی نیست.")
        st.stop()

    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))

    # بیمه آپشن پوت
    use_put_option = st.checkbox("فعال‌سازی بیمه با آپشن پوت")
    if use_put_option:
        insurance_percent = st.number_input(
            "درصد پوشش بیمه (٪)", min_value=0.0, max_value=100.0, value=30.0
        )
        adjusted_cov = cov_matrix * (1 - insurance_percent / 100) ** 2
    else:
        adjusted_cov = cov_matrix

    n_portfolios = 5000
    n_assets = len(asset_names)
    results = np.zeros((3 + n_assets, n_portfolios))

    np.random.seed(42)
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))
        sharpe = port_return / port_std
        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe
        results[3:, i] = weights

    # بهترین پرتفو نزدیک به ریسک هدف
    target_risk = st.slider("ریسک هدف سالانه (انحراف معیار)", 0.01, 1.0, 0.25, 0.01)
    idx = np.argmin(np.abs(results[1] - target_risk))
    best_return = results[0, idx]
    best_std = results[1, idx]
    best_sharpe = results[2, idx]
    best_weights = results[3:, idx]

    st.subheader("🔍 پرتفو پیشنهادی")
    st.write(f"بازده سالانه: {best_return:.2%}")
    st.write(f"ریسک سالانه: {best_std:.2%}")
    st.write(f"نسبت شارپ: {best_sharpe:.2f}")
    for i, asset in enumerate(asset_names):
        st.write(f"{asset}: {best_weights[i]*100:.2f}%")

    # نمودار پراکندگی
    fig = px.scatter(
        x=results[1]*100,
        y=results[0]*100,
        color=results[2],
        labels={'x': 'ریسک سالانه (%)', 'y': 'بازده سالانه (%)'},
        title='شبیه‌سازی پرتفوها',
        color_continuous_scale='Viridis',
        width=800, height=500
    )
    fig.add_scatter(
        x=[best_std*100], y=[best_return*100],
        mode='markers', marker=dict(color='red', size=15, symbol='star'),
        name='پرتفو بهینه'
    )
    st.plotly_chart(fig)

    # سود و زیان دلاری
    st.subheader("💰 محاسبه سود و زیان تخمینی (دلار)")
    base_amount = st.number_input("مقدار دارایی پایه (واحد)", min_value=0.0, value=1.0)
    base_price = st.number_input("قیمت پایه هر واحد (دلار)", min_value=0.0, value=1000.0)

    est_profit = base_amount * base_price * best_return
    est_loss = base_amount * base_price * best_std

    st.write(f"سود تخمینی: {est_profit:,.2f} دلار")
    st.write(f"ضرر تخمینی (انحراف معیار): {est_loss:,.2f} دلار")

    # احتمال بازده در ±۱ انحراف معیار
    conf_level = 0.68
    z = norm.ppf((1 + conf_level) / 2)
    lower_bound = best_return - z * best_std
    upper_bound = best_return + z * best_std
    st.write(f"احتمال بازده در بازه ±۱ انحراف معیار ({int(conf_level*100)}%):")
    st.write(f"{lower_bound:.2%} تا {upper_bound:.2%}")

    # احتمال دراگون (افت شدید)
    st.subheader("⚠️ احتمال افت شدید (دراگون)")
    threshold = st.number_input("آستانه افت (مثلاً 0 یا -10٪)", value=0.0, step=0.01)
    prob_dragon = norm.cdf(threshold, loc=best_return, scale=best_std)
    st.write(f"احتمال بازده کمتر از {threshold:.2%}: {prob_dragon*100:.2f}%")

else:
    st.info("لطفاً حداقل یک فایل CSV با ستون Date و Price آپلود کنید.")
