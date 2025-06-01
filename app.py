import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from io import StringIO

# پیکربندی صفحه
st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو و استراتژی Married Put")

# 🧾 دکمه‌های دانلود فایل‌های نمونه و بارگذاری مستقیم
st.sidebar.markdown("---")
st.sidebar.subheader("📥 داده‌های نمونه و بارگذاری آنی")

sample_files = {
    "📊 BTC/USD نمونه 7 ساله (هفتگی)": "https://raw.githubusercontent.com/mohammadmarghzari/monte-carlo-portfolio/main/data/BTC_USD%207%20Years%20Weekly.csv",
    "📉 ETH/USD نمونه 7 ساله (هفتگی)": "https://raw.githubusercontent.com/mohammadmarghzari/monte-carlo-portfolio/main/data/ETH_USD%207%20Years%20Weekly.csv"
}

loaded_samples = {}

for label, url in sample_files.items():
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filename = url.split("/")[-1]
            st.sidebar.download_button(label, data=response.content, file_name=filename)
            if st.sidebar.button(f"📂 بارگذاری {label}", key=f"load_{label}"):
                file_content = StringIO(response.text)
                df = pd.read_csv(file_content)
                # استانداردسازی نام ستون‌ها به حروف کوچک
                df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
                df.rename(columns={'date': 'date', 'price': 'price'}, inplace=True)
                loaded_samples[filename.split('.')[0]] = df
        else:
            st.sidebar.warning(f"❌ خطا در بارگیری فایل {label}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ مشکل در اتصال برای {label}: {e}")

# تابع خواندن فایل CSV
def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
        df.rename(columns={'date': 'date', 'price': 'price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {getattr(file, 'name', 'بدون نام')}: {e}")
        return None

# بارگذاری فایل‌ها
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True)

# ادغام فایل‌های نمونه اگر آپلود دستی انجام نشد
if not uploaded_files and loaded_samples:
    uploaded_files = []
    for name, df in loaded_samples.items():
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        csv_buffer.name = f"{name}.csv"  # اضافه کردن ویژگی name به صورت دستی
        uploaded_files.append(csv_buffer)

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

        name = getattr(file, 'name', 'Asset').split('.')[0]

        if 'date' not in df.columns or 'price' not in df.columns:
            st.warning(f"فایل {name} باید دارای ستون‌های 'date' و 'price' باشد.")
            continue

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['price'] = df['price'].astype(str).str.replace(',', '')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['date', 'price'])
        df = df[['date', 'price']].set_index('date')
        df.columns = [name]

        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')  # یا 'outer' به دلخواه شما
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
