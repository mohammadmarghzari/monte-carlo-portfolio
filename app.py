import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="تحلیل پرتفو با بیمه آپشن، مونت‌کارلو و مرز کارا", layout="wide")
st.title("📈 تحلیل پرتفو با بیمه آپشن، مونت‌کارلو و مرز کارا")
st.markdown("محاسبه دقیق سود و ریسک با محاسبه جداگانه آپشن برای هر دارایی")

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
        name = file.name.split('.')[0]
        df = pd.read_csv(file)

        # پاکسازی نام ستون‌ها: حذف کوتیشن، اسپیس و تبدیل به حروف کوچک
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", '').str.lower()

        # اطمینان از وجود ستون‌های 'date' و 'price'
        if 'date' not in df.columns or 'price' not in df.columns:
            st.error(f"فایل {name} باید ستون‌های 'Date' و 'Price' داشته باشد. ستون‌های یافت شده: {df.columns.tolist()}")
            continue

        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)

        # فقط ستون Date و Price را نگه دارید
        df = df[['Date', 'Price']].copy()

        # حذف ردیف‌های با مقدار ناقص
        df.dropna(subset=['Date', 'Price'], inplace=True)

        # تبدیل ستون تاریخ به datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # حذف ردیف‌هایی که تاریخ آنها نامعتبر است
        df = df.dropna(subset=['Date'])

        # تنظیم Date به عنوان ایندکس
        df.set_index('Date', inplace=True)

        # فقط ستون Price باقی می‌ماند
        df = df[['Price']]

        # بررسی اینکه فقط یک ستون باقی مانده باشد
        if df.shape[1] != 1:
            st.error(f"خطا: فایل {name} ستون Price اضافی یا تکراری دارد.")
            continue

        # تغییر نام ستون Price به اسم دارایی
        df.columns = [name]

        if prices_df.empty:
            prices_df = df
        else:
            prices_df = prices_df.join(df, how='inner')

        asset_names.append(name)

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    # ادامه کد شما: محاسبه بازده، کوواریانس، بیمه آپشن و شبیه‌سازی پرتفو و ...

    # --- نمونه نمایش داده‌ها ---
    st.subheader("🧪 پیش‌نمایش داده‌ها (آخرین قیمت‌ها)")
    st.dataframe(prices_df.tail())

    # محاسبه بازده و کوواریانس
    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()

    if returns.empty:
        st.error("❌ محاسبه بازده ممکن نیست، داده‌ها را بررسی کنید.")
        st.stop()

    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    asset_std_devs = np.sqrt(np.diag(cov_matrix))

    # بقیه کد تحلیل و نمایش

else:
    st.warning("لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید.")
