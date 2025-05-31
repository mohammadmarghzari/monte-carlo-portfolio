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
    "📊 BTC/USD نمونه 7 ساله (هفتگی)": "https://raw.githubusercontent.com/USERNAME/REPO/main/data/BTC_USD%207%20Years%20Weekly.csv",
    "📉 ETH/USD نمونه 7 ساله (هفتگی)": "https://raw.githubusercontent.com/USERNAME/REPO/main/data/ETH_USD%207%20Years%20Weekly.csv"
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
                df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
                df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
                loaded_samples[filename.split('.')[0]] = df
        else:
            st.sidebar.warning(f"❌ خطا در بارگیری فایل {label}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ مشکل در اتصال برای {label}")

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

# ادغام فایل‌های نمونه اگر آپلود دستی انجام نشد
if not uploaded_files and loaded_samples:
    uploaded_files = []
    for name, df in loaded_samples.items():
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        uploaded_files.append(csv_buffer)
        uploaded_files[-1].name = f"{name}.csv"

# باقی کد بدون تغییر ادامه پیدا می‌کند (استفاده از uploaded_files)
