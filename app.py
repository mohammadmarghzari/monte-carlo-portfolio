# ابزار کامل تحلیل پرتفو با پشتیبانی از بیمه دارایی و MPT
# توسعه یافته بر اساس نیاز شما توسط ChatGPT

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime, timedelta

st.set_page_config(page_title="تحلیل پرتفو پیشرفته با بیمه و MPT", layout="wide")
st.title("📊 تحلیل پرتفو با بیمه دارایی‌ها و مرز کارای MPT")

# ---------------------- تابع خواندن فایل ----------------------
def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.replace('%', '').str.lower()
        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

# ---------------------- بارگذاری داده ----------------------
st.sidebar.header("📂 بارگذاری فایل‌ها")
uploaded_files = st.sidebar.file_uploader("فایل‌های CSV شامل ستون‌های Date و Price", type=['csv'], accept_multiple_files=True)

period = st.sidebar.selectbox("بازه تحلیل بازده:", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []
    insurance_data = {}

    st.subheader("⚙️ تنظیمات بیمه برای هر دارایی")

    for file in uploaded_files:
        df = read_csv_file(file)
        if df is None:
            continue

        name = file.name.split('.')[0]

        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"فایل {name} فاقد ستون‌های لازم است.")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]

        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

        with st.expander(f"🔒 بیمه برای {name}"):
            insured = st.checkbox(f"فعال‌سازی بیمه برای {name}", key=name)
            if insured:
                base_price = st.number_input("قیمت دارایی پایه", min_value=0.0, value=100.0, step=0.01, format="%.2f", key=f"base_{name}")
                base_amount = st.number_input("مقدار دارایی پایه", min_value=0.0, value=1.0, step=0.01, format="%.2f", key=f"amount_{name}")
                strike_price = st.number_input("قیمت اعمال آپشن پوت", min_value=0.0, value=90.0, step=0.01, format="%.2f", key=f"strike_{name}")
                premium = st.number_input("قیمت قرارداد (پرمیوم)", min_value=0.0, value=2.0, step=0.01, format="%.2f", key=f"premium_{name}")
                maturity = st.date_input("تاریخ سررسید", value=datetime.today() + timedelta(days=30), key=f"maturity_{name}")
                contract_amount = st.number_input("تعداد قرارداد", min_value=0.0, value=1.0, step=0.01, format="%.2f", key=f"contract_{name}")
                loss_covered = st.number_input("درصد پوشش ضرر", min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.3f", key=f"loss_{name}")

                insurance_data[name] = {
                    'insured': True,
                    'base_price': base_price,
                    'base_amount': base_amount,
                    'strike': strike_price,
                    'premium': premium,
                    'maturity': maturity,
                    'contracts': contract_amount,
                    'loss_covered': loss_covered
                }

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری یافت نشد.")
        st.stop()

    st.subheader("📊 پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail())

    resampled = prices_df.resample(resample_rule).last().dropna()
    returns = resampled.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor

    asset_std = np.sqrt(np.diag(cov_matrix))

    # ---------------------- تنظیم وزن‌ها با در نظر گرفتن بیمه ----------------------
    effective_std = asset_std.copy()
    for i, name in enumerate(asset_names):
        if name in insurance_data:
            loss_cov = insurance_data[name]['loss_covered']
            effective_std[i] *= (1 - loss_cov)

    preference_weights = asset_std / effective_std
    preference_weights /= preference_weights.sum()

    # ---------------------- مونت کارلو ----------------------
    n_portfolios = 10000
    n_assets = len(asset_names)
    results = np.zeros((3 + n_assets, n_portfolios))

    np.random.seed(42)
    for i in range(n_portfolios):
        rand = np.random.random(n_assets)
        weights = rand * preference_weights
        weights /= weights.sum()
        ret = np.dot(weights, mean_returns)
        risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = ret / risk
        results[0, i] = ret
        results[1, i] = risk
        results[2, i] = sharpe
        results[3:, i] = weights

    # ---------------------- مرز کارا (MPT) ----------------------
    st.subheader("📐 مرز کارای پرتفو (MPT)")
    fig_mpt = go.Figure()
    fig_mpt.add_trace(go.Scatter(
        x=results[1]*100, y=results[0]*100, mode='markers', marker=dict(color=results[2], colorscale='Viridis', showscale=True),
        name="Portfolios"
    ))
    st.plotly_chart(fig_mpt, use_container_width=True)

    # ---------------------- نمودار سود/زیان بیمه ----------------------
    st.subheader("🧮 نمودار سود و زیان بیمه‌ها")
    for name, info in insurance_data.items():
        x_vals = np.linspace(0.5 * info['base_price'], 1.5 * info['base_price'], 100)
        profit = np.where(x_vals < info['strike'],
                          (info['strike'] - x_vals) * info['contracts'] - info['premium'] * info['contracts'],
                          -info['premium'] * info['contracts'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_vals, y=profit, mode='lines', name="سود/زیان آپشن"))
        fig.update_layout(title=f"سود و زیان بیمه برای {name}", xaxis_title="قیمت پایانی دارایی", yaxis_title="سود/زیان", height=400)
        st.plotly_chart(fig, use_container_width=True)

        img_bytes = fig.to_image(format="png")
        st.download_button(
            label=f"📥 دانلود نمودار {name} به‌صورت عکس",
            data=img_bytes,
            file_name=f"option_profit_{name}.png",
            mime="image/png"
        )
