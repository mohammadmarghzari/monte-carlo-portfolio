import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import chardet
import csv
import io

# تنظیمات صفحه
st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با روش مونت‌کارلو")
st.markdown("ریسک هر دارایی = ۲۰٪ | هدف: ساخت پرتفو با ریسک نزدیک به ۳۰٪")

# تابع تشخیص رمزگذاری
def detect_encoding(file):
    try:
        raw_data = file.read(10000)
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'
    except Exception as e:
        st.error(f"خطا در تشخیص رمزگذاری: {e}")
        return 'utf-8'

# تابع تشخیص جداکننده
def detect_delimiter(file, encoding='utf-8'):
    try:
        file.seek(0)
        sample = file.readline().decode(encoding)
        sniffer = csv.Sniffer()
        return sniffer.sniff(sample).delimiter
    except Exception as e:
        st.error(f"خطا در تشخیص جداکننده: {e}")
        return ','

# تابع خواندن فایل CSV
def read_csv_file(file):
    try:
        encoding = detect_encoding(file)
        st.info(f"رمزگذاری فایل {file.name}: {encoding}")
        delimiter = detect_delimiter(file, encoding)
        st.info(f"جداکننده فایل: {delimiter}")
        
        file.seek(0)
        df = pd.read_csv(
            file,
            encoding=encoding,
            sep=delimiter,
            decimal='.',
            thousands=None,
            na_values=['', 'NA', 'N/A', 'null'],
            skipinitialspace=True,
            on_bad_lines='warn'
        )
        return df
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
    "چند Ascending چند فایل CSV آپلود کنید (هر دارایی یک فایل)",
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

    # انتخاب پرتفو با ریسک نزدیک به ۳۰٪
    target_risk = 0.30
    best_idx = np.argmin(np.abs(results[1] - target_risk))

    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_weights = results[3:, best_idx]

    # نمایش نتایج
    st.subheader("📊 نتایج پرتفو پیشنهادی")
    st.markdown(f"""
    - ✅ **بازده مورد انتظار سالانه:** {best_return:.2%}  
    - ⚠️ **ریسک سالانه:** {best_risk:.2%}  
    - 🧠 **نسبت شارپ:** {best_sharpe:.2f}  
    """)

    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 **وزن {name}:** {best_weights[i]*100:.2f}٪")

    # نمودار سود/زیان
    st.subheader("📈 نمودار سود/زیان پرتفو نسبت به تغییر قیمت‌ها")
    price_changes = np.linspace(-0.5, 0.5, 100)
    total_change = np.zeros_like(price_changes)

    for i, w in enumerate(best_weights):
        total_change += w * price_changes

    plt.figure(figsize=(8, 4))
    plt.plot(price_changes * 100, total_change * 100, label="تغییر ارزش پرتفو")
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("درصد تغییر قیمت دارایی‌ها")
    plt.ylabel("درصد سود/زیان پرتفو")
    plt.title("نمودار سود/زیان پرتفو")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

else:
    st.warning("لطفاً چند فایل CSV قیمت دارایی‌ها آپلود کنید.")
