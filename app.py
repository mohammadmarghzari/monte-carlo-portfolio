import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# پیکربندی صفحه
st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو و استراتژی Married Put")

# تابع خواندن فایل CSV
def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

# بارگذاری فایل‌ها
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True)

# تنظیمات بازه زمانی
period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []
    insured_assets = {}

    for file in uploaded_files:
        df = read_csv_file(file)
        if df is None:
            continue

        name = file.name.split('.')[0]

        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"فایل {name} باید دارای ستون‌های 'Date' و 'Price' باشد.")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]

        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

        # تنظیمات بیمه در سایدبار برای هر دارایی
        st.sidebar.markdown(f"---\n### ⚙️ تنظیمات بیمه برای دارایی: `{name}`")
        insured = st.sidebar.checkbox(f"📌 فعال‌سازی بیمه برای {name}", key=f"insured_{name}")
        if insured:
            loss_percent = st.sidebar.number_input(f"📉 درصد ضرر معامله پوت برای {name}", 0.0, 100.0, 30.0, step=0.01, key=f"loss_{name}")
            strike = st.sidebar.number_input(f"🎯 قیمت اعمال پوت برای {name}", 0.0, 1e6, 100.0, step=0.01, key=f"strike_{name}")
            premium = st.sidebar.number_input(f"💰 قیمت قرارداد پوت برای {name}", 0.0, 1e6, 5.0, step=0.01, key=f"premium_{name}")
            amount = st.sidebar.number_input(f"📦 مقدار قرارداد برای {name}", 0.0, 1e6, 1.0, step=0.01, key=f"amount_{name}")
            spot_price = st.sidebar.number_input(f"📌 قیمت فعلی دارایی پایه {name}", 0.0, 1e6, 100.0, step=0.01, key=f"spot_{name}")
            asset_amount = st.sidebar.number_input(f"📦 مقدار دارایی پایه {name}", 0.0, 1e6, 1.0, step=0.01, key=f"base_{name}")
            insured_assets[name] = {
                'loss_percent': loss_percent,
                'strike': strike,
                'premium': premium,
                'amount': amount,
                'spot': spot_price,
                'base': asset_amount
            }

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    st.subheader("🧪 پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail())

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()

    mean_returns = returns.mean() * annual_factor

    # 1. دریافت ریسک سالانه هر دارایی از کاربر (پیش‌فرض 20%)
    st.sidebar.header("⚙️ تنظیمات ریسک دارایی‌ها")
    asset_risks = {}
    for name in asset_names:
        risk = st.sidebar.number_input(
            f"ریسک سالانه دارایی {name} (%)",
            min_value=0.0, max_value=100.0, value=20.0, step=0.1, key=f"risk_{name}"
        )
        asset_risks[name] = risk / 100  # درصد به عدد اعشاری

    # 2. ریسک هدف پورتفو
    target_risk = st.sidebar.number_input(
        "ریسک هدف پورتفو (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1
    ) / 100

    # 3. ساخت ماتریس کوواریانس با ریسک‌های ورودی و حفظ همبستگی واقعی
    correlation_matrix = returns.corr()
    cov_matrix_fixed = np.zeros_like(correlation_matrix.values)
    for i, name_i in enumerate(asset_names):
        for j, name_j in enumerate(asset_names):
            if i == j:
                cov_matrix_fixed[i, j] = asset_risks[name_i] ** 2
            else:
                cov_matrix_fixed[i, j] = correlation_matrix.iloc[i, j] * asset_risks[name_i] * asset_risks[name_j]

    # محاسبه انحراف معیار دارایی‌ها
    std_devs = np.sqrt(np.diag(cov_matrix_fixed))

    # وزن‌دهی ترجیحی (برعکس ریسک هر دارایی)
    preference_weights = 1 / std_devs
    preference_weights /= np.sum(preference_weights)

    # شبیه‌سازی مونت‌کارلو
    n_portfolios = 10000
    results = np.zeros((3 + len(asset_names), n_portfolios))
    np.random.seed(42)

    for i in range(n_portfolios):
        weights = np.random.random(len(asset_names)) * preference_weights
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_fixed, weights)))
        sharpe_ratio = port_return / port_std if port_std != 0 else 0
        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3:, i] = weights

    # انتخاب بهترین پورتفو بر اساس ریسک هدف
    best_idx = np.argmin(np.abs(results[1] - target_risk))
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_weights = results[3:, best_idx]

    st.subheader("📈 پورتفو بهینه")
    st.markdown(f"""
    - ✅ بازده سالانه: **{best_return:.2%}**
    - ⚠️ ریسک سالانه: **{best_risk:.2%}**
    - 🧠 نسبت شارپ: **{best_sharpe:.2f}**
    """)

    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 وزن {name}: {best_weights[i]*100:.2f}%")

    # نمودار پراکندگی ریسک-بازده
    fig = px.scatter(x=results[1]*100, y=results[0]*100, color=results[2],
                     labels={'x': 'ریسک (%)', 'y': 'بازده (%)'},
                     title='پرتفوهای شبیه‌سازی‌شده', color_continuous_scale='Viridis')
    fig.add_trace(go.Scatter(x=[best_risk*100], y=[best_return*100],
                             mode='markers', marker=dict(size=12, color='red', symbol='star'),
                             name='پرتفوی بهینه'))
    st.plotly_chart(fig, key="monte_carlo_scatter")

    # نمودار Married Put برای هر دارایی بیمه شده
    for name, info in insured_assets.items():
        st.subheader(f"📉 نمودار سود و زیان استراتژی Married Put - {name}")
        x = np.linspace(info['spot'] * 0.5, info['spot'] * 1.5, 200)
        asset_pnl = (x - info['spot']) * info['base']
        put_pnl = np.where(x < info['strike'], (info['strike'] - x) * info['amount'], 0) - info['premium'] * info['amount']
        total_pnl = asset_pnl + put_pnl

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=total_pnl, mode='lines', name='Married Put'))
        fig2.add_trace(go.Scatter(x=x, y=asset_pnl, mode='lines', name='دارایی پایه', line=dict(dash='dot')))
        fig2.add_trace(go.Scatter(x=x, y=put_pnl, mode='lines', name='پوت', line=dict(dash='dot')))
        fig2.update_layout(title='نمودار سود و زیان', xaxis_title='قیمت دارایی در سررسید', yaxis_title='سود/زیان')

        st.plotly_chart(fig2, key=f"married_put_chart_{name}")

        if st.button(f"📷 ذخیره نمودار Married Put برای {name}", key=f"save_{name}"):
            fig2.write_image(f"married_put_{name}.png")
            st.success(f"نمودار Married Put برای {name} ذخیره شد.")

else:
    st.info("لطفاً حداقل یک فایل CSV آپلود کنید.")

