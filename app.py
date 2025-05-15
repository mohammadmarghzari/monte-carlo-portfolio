import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="تحلیل پرتفو با آپشن و مونت‌کارلو + مرز کارا", layout="wide")
st.title("📈 تحلیل پرتفو با بیمه آپشن، مونت‌کارلو و مرز کارا")
st.markdown("هدف: محاسبه دقیق بازده و ریسک با پوشش آپشن پوت")

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

analysis_mode = st.sidebar.radio("روش تحلیل پرتفو:", ["مونت‌کارلو (MC)", "مرز کارا (MPT)"])

period = st.sidebar.selectbox("بازه تحلیل بازده", ['روزانه', 'ماهانه', 'سه‌ماهه'])
if period == 'روزانه':
    resample_rule = 'D'
    annual_factor = 252
elif period == 'ماهانه':
    resample_rule = 'M'
    annual_factor = 12
else:
    resample_rule = 'Q'
    annual_factor = 4

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []

    for file in uploaded_files:
        df = read_csv_file(file)
        if df is None:
            continue

        name = file.name.split('.')[0]

        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"فایل {name} باید ستون‌های 'Date' و 'Price' داشته باشد. ستون‌های یافت‌شده: {df.columns.tolist()}")
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

    st.subheader("🧪 پیش‌نمایش داده‌ها (آخرین قیمت‌ها)")
    st.dataframe(prices_df.tail())

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()

    if returns.empty:
        st.error("❌ محاسبه بازده ممکن نیست، داده‌ها را بررسی کنید.")
        st.stop()

    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor

    asset_std_devs = np.sqrt(np.diag(cov_matrix))

    # ورودی‌های بیمه آپشن
    st.sidebar.subheader("بیمه آپشن پوت")
    use_put_option = st.sidebar.checkbox("فعال‌سازی بیمه با آپشن پوت")

    if use_put_option:
        insurance_percent = st.sidebar.number_input("درصد پوشش بیمه (٪ از پرتفو)", min_value=0.0, max_value=100.0, value=30.0)
        option_strike_price = st.sidebar.number_input("قیمت اعمال (Strike Price) آپشن", min_value=0.0, value=1000.0)
        option_premium = st.sidebar.number_input("قیمت آپشن (Premium)", min_value=0.0, value=50.0)
        option_contracts = st.sidebar.number_input("تعداد قرارداد آپشن", min_value=0, value=1)

        # ورودی دارایی پایه برای محاسبه پوشش واقعی
        base_amount = st.number_input("مقدار دارایی پایه (تعداد واحد)", min_value=0.0, value=1.0, step=0.01)
        base_price_usd = st.number_input("قیمت پایه دلاری هر واحد دارایی", min_value=0.0, value=1000.0, step=0.01)

        total_value_usd = base_amount * base_price_usd

        # محاسبه پوشش پوشش بیمه بر اساس قیمت اعمال و تعداد قرارداد آپشن
        # فرض: هر قرارداد آپشن معادل ۱ واحد دارایی پایه (اگر لازم بود عددش رو تنظیم کن)
        insurance_coverage_value = option_contracts * option_strike_price

        # درصد پوشش واقعی روی کل پرتفوی
        real_coverage_percent = min(insurance_coverage_value / total_value_usd, 1.0)

        st.write(f"درصد واقعی پوشش بیمه شده از پرتفو: {real_coverage_percent*100:.2f}%")

        # هزینه کل بیمه (Premium * تعداد قرارداد)
        total_premium_cost = option_premium * option_contracts

        # کاهش بازده پرتفوی بخاطر هزینه بیمه
        adjusted_mean_returns = mean_returns * (1 - real_coverage_percent) - total_premium_cost / total_value_usd

        # کاهش کوواریانس (ریسک) به دلیل بیمه آپشن (کاهش ریسک به میزان پوشش واقعی)
        adjusted_cov = cov_matrix * (1 - real_coverage_percent) ** 2

        # وزن‌دهی ترجیحی: وقتی بیمه هست، وزن دارایی‌های ریسک بالاتر بیشتر می‌شود
        effective_std = asset_std_devs * (1 - real_coverage_percent)
        preference_weights = effective_std / asset_std_devs
        preference_weights /= np.sum(preference_weights)

    else:
        adjusted_mean_returns = mean_returns
        adjusted_cov = cov_matrix
        preference_weights = 1 / asset_std_devs
        preference_weights /= np.sum(preference_weights)

    np.random.seed(42)

    if analysis_mode == "مونت‌کارلو (MC)":
        st.header("🔁 تحلیل با شبیه‌سازی مونت‌کارلو")
        n_portfolios = 10000
        n_assets = len(asset_names)
        results = np.zeros((3 + n_assets, n_portfolios))

        for i in range(n_portfolios):
            random_factors = np.random.random(n_assets)
            weights = random_factors * preference_weights
            weights /= np.sum(weights)
            port_return = np.dot(weights, adjusted_mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))
            sharpe_ratio = port_return / port_std if port_std != 0 else 0
            results[0, i] = port_return
            results[1, i] = port_std
            results[2, i] = sharpe_ratio
            results[3:, i] = weights

        target_risk = 0.25
        best_idx = np.argmin(np.abs(results[1] - target_risk))
        best_return = results[0, best_idx]
        best_risk = results[1, best_idx]
        best_sharpe = results[2, best_idx]
        best_weights = results[3:, best_idx]

        st.markdown(f"""
        - ✅ بازده مورد انتظار: {best_return:.2%}
        - ⚠️ ریسک (انحراف معیار): {best_risk:.2%}
        - 🧠 نسبت شارپ: {best_sharpe:.2f}
        """)
        for i, name in enumerate(asset_names):
            st.markdown(f"🔹 وزن {name}: {best_weights[i]*100:.2f}%")

        fig = px.scatter(
            x=results[1]*100,
            y=results[0]*100,
            color=results[2],
            labels={'x': 'Risk (%)', 'y': 'Expected Return (%)'},
            title="Efficient Frontier - Monte Carlo",
            color_continuous_scale='Viridis'
        )
        fig.add_trace(go.Scatter(
            x=[best_risk * 100],
            y=[best_return * 100],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='Target Portfolio'
        ))
        st.plotly_chart(fig)

    else:
        st.header("📈 تحلیل با مرز کارا (MPT)")
        n_points = 500
        n_assets = len(asset_names)
        results = np.zeros((3 + n_assets, n_points))

        for i in range(n_points):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            port_return = np.dot(weights, adjusted_mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))
            sharpe = port_return / port_std if port_std != 0 else 0
            results[0, i] = port_return
            results[1, i] = port_std
            results[2, i] = sharpe
            results[3:, i] = weights

        target_risk = 0.25
        idx_target = np.argmin(np.abs(results[1] - target_risk))
        r_return = results[0, idx_target]
        r_risk = results[1, idx_target]
        r_weights = results[3:, idx_target]

        st.markdown(f"""
        - ✅ بازده برای ریسک 25٪: {r_return:.2%}
        - ⚠️ ریسک: {r_risk:.2%}
        """)
        for i, name in enumerate(asset_names):
            st.markdown(f"🔸 وزن {name}: {r_weights[i]*100:.2f}%")

        fig = px.scatter(
            x=results[1]*100,
            y=results[0]*100,
            color=results[2],
            labels={'x': 'Risk (%)', 'y': 'Expected Return (%)'},
            title="Efficient Frontier - MPT",
            color_continuous_scale='Turbo'
        )
        fig.add_trace(go.Scatter(
            x=[r_risk*100],
            y=[r_return*100],
            mode='markers',
            marker=dict(color='red', size=12, symbol='star'),
            name='Target Portfolio'
        ))
        st.plotly_chart(fig)

    st.subheader("💰 محاسبه سود و زیان تخمینی (دلار آمریکا)")
    total_value_usd = base_amount * base_price_usd

    # انتخاب بازده و ریسک مطابق روش تحلیل
    if analysis_mode == "مونت‌کارلو (MC)":
        selected_return = best_return
        selected_risk = best_risk
        selected_weights = best_weights
    else:
        selected_return = r_return
        selected_risk = r_risk
        selected_weights = r_weights

    estimated_profit_usd = total_value_usd * selected_return
    estimated_loss_usd = total_value_usd * selected_risk

    st.markdown(f"""
    - 📈 سود تخمینی: {estimated_profit_usd:,.2f} دلار  
    - 📉 ضرر تخمینی (انحراف معیار): {estimated_loss_usd:,.2f} دلار  
    """)

    st.markdown("### 🎯 بازده مورد انتظار سالانه:")
    st.write(f"{selected_return:.2%}")

    st.markdown("### 📊 ریسک سالانه (انحراف معیار):")
    st.write(f"{selected_risk:.2%}")

else:
    st.warning("لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید.")
