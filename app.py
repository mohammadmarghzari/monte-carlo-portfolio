import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# تنظیمات صفحه
st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با روش مونت‌کارلو")
st.markdown("ریسک هر دارایی = ۲۰٪ | هدف: ساخت پرتفو با ریسک نزدیک به ۳۰٪")

# تابع خواندن فایل CSV
def read_csv_file(file):
    try:
        df = pd.read_csv(
            file,
            encoding='utf-8',
            sep=',',  # فرض جداکننده کاما
            decimal='.',
            thousands=None,
            na_values=['', 'NA', 'N/A', 'null'],
            skipinitialspace=True,
            on_bad_lines='warn'
        )
        return df
    except UnicodeDecodeError:
        st.error(f"خطا در رمزگذاری فایل {file.name}. لطفاً فایل را با رمزگذاری UTF-8 ذخیره کنید.")
        return None
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

# تابع یافتن ستون قیمت
def find_price_column(df, file_name):
    possible_cols = [
        col for col in df.columns 
        if any(key in col.lower() for key in ['price', 'close', 'adj close', 'adjusted close'])
    ]
    if possible_cols:
        return possible_cols[0]
    st.warning(f"ستونی مشابه 'Price' یا 'Close' در فایل {file_name} یافت نشد.")
    return None

# سایدبار برای آپلود فایل‌ها
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)",
    type=['csv'],
    accept_multiple_files=True
)

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []
    date_column = None

    # پردازش فایل‌های آپلودشده
    for file in uploaded_files:
        df = read_csv_file(file)
        if df is None:
            continue
            
        name = file.name.split('.')[0]
        st.write(f"📄 فایل: {name} - ستون‌ها: {list(df.columns)}")

        # یافتن ستون تاریخ
        possible_date_cols = [col for col in df.columns if 'date' in col.lower()]
        if not possible_date_cols:
            st.error(f"❌ فایل '{name}' فاقد ستون تاریخ است.")
            continue
        date_col = possible_date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        if df[date_col].isna().any():
            st.error(f"❌ برخی مقادیر تاریخ در فایل '{name}' نامعتبر هستند.")
            continue

        # یافتن یا انتخاب ستون قیمت
        close_col = find_price_column(df, name)
        if not close_col:
            st.write("لطفاً ستون قیمت را به‌صورت دستی انتخاب کنید:")
            close_col = st.selectbox(
                f"انتخاب ستون قیمت برای {name}",
                options=df.columns,
                key=f"price_col_{name}"
            )
        
        # پاکسازی داده‌های قیمت
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        if df[close_col].isna().any():
            st.warning(f"⚠️ {df[close_col].isna().sum()} مقدار نامعتبر در ستون '{close_col}' فایل '{name}' یافت شد.")
            df = df.dropna(subset=[close_col])
        
        # تنظیم تاریخ به‌عنوان شاخص و ادغام داده‌ها
        df = df[[date_col, close_col]].set_index(date_col)
        df.columns = [name]
        if prices_df.empty:
            prices_df = df
            date_column = date_col
        else:
            prices_df = prices_df.join(df, how='inner')

        asset_names.append(name)
        st.success(f"✅ ستون انتخاب‌شده برای {name}: {close_col}")

    if prices_df.empty or len(asset_names) < 1:
        st.error("❌ هیچ داده معتبری از فایل‌ها استخراج نشد.")
        st.stop()

    # محاسبه بازده و کوواریانس
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    # شبیه‌سازی مونت‌کارلو
