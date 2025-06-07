import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import base64

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو، CVaR و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو، CVaR و استراتژی Married Put")

def read_csv_file(file):
    try:
        df = pd.read_csv(file, header=None)
        if df.shape[0] < 2:
            raise Exception("تعداد سطرهای داده بسیار کم است.")
        # پیدا کردن سطر عنوان (ستونی که شامل چیزی شبیه date باشد)
        header_idx = None
        for i in range(min(5, len(df))):
            row = [str(x).strip().lower() for x in df.iloc[i].tolist()]
            if any('date' == x for x in row):
                header_idx = i
                break
        if header_idx is None:
            raise Exception("سطر عنوان مناسب (شامل date) یافت نشد.")
        header_row = df.iloc[header_idx].tolist()
        df = df.iloc[header_idx+1:].reset_index(drop=True)
        df.columns = header_row

        # پیدا کردن نام ستون تاریخ با حساسیت پایین
        date_col = [c for c in df.columns if str(c).strip().lower() == 'date']
        if not date_col:
            raise Exception("ستون تاریخ با نام 'Date' یا مشابه آن یافت نشد.")
        date_col = date_col[0]

        # پیدا کردن یک ستون قیمت مناسب (Case-insensitive)
        price_candidates = [c for c in df.columns if str(c).strip().lower() in ['price', 'close', 'adj close', 'open']]
        if not price_candidates:
            price_candidates = [c for c in df.columns if c != date_col]
        if not price_candidates:
            raise Exception("ستون قیمت مناسب یافت نشد.")
        price_col = price_candidates[0]

        # فقط دو ستون انتخاب شود و هیچ کدام خالی نباشد
        df = df[[date_col, price_col]].dropna()
        if df.empty:
            raise Exception("پس از حذف داده‌های خالی، داده‌ای باقی نماند.")

        df = df.rename(columns={date_col: "Date", price_col: "Price"})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        if df.empty:
            raise Exception("پس از تبدیل نوع داده، داده معتبری باقی نماند.")

        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

def download_link(df, filename):
    csv = df.reset_index(drop=True).to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ دریافت فایل CSV</a>'

def get_price_dataframe_from_yf(data, t):
    # اگر دیتا چند نماد است:
    if isinstance(data.columns, pd.MultiIndex):
        # اگر برای نماد t وجود دارد
        if t in data.columns.levels[0]:
            df_t = data[t].reset_index()
            price_col = None
            for col in ['Close', 'Adj Close', 'Open']:
                if col in df_t.columns:
                    price_col = col
                    break
            if price_col is None:
                return None, f"هیچ یک از ستون‌های قیمت (Close, Adj Close, Open) برای {t} پیدا نشد."
            df = df_t[['Date', price_col]].rename(columns={price_col: 'Price'})
            return df, None
        else:
            return None, f"نماد {t} در داده‌های دریافتی وجود ندارد."
    else:
        # دیتا فقط برای یک نماد است
        if 'Date' not in data.columns:
            data = data.reset_index()
        price_col = None
        for col in ['Close', 'Adj Close', 'Open']:
            if col in data.columns:
                price_col = col
                break
        if price_col is None:
            return None, f"هیچ یک از ستون‌های قیمت (Close, Adj Close, Open) برای {t} پیدا نشد."
        df = data[['Date', price_col]].rename(columns={price_col: 'Price'})
        return df, None

with st.expander("📘 راهنمای دریافت داده آنلاین از یاهو فاینانس"):
    st.markdown("""
    <div dir="rtl" style="text-align: right; font-size: 15px">
    <b>نحوه دانلود داده:</b><br>
    - نماد هر دارایی را مطابق سایت Yahoo Finance وارد کنید.<br>
    - نمادها را با <b>کاما</b> و <b>بدون فاصله</b> وارد کنید.<br>
    - مثال: <b>BTC-USD,AAPL,ETH-USD</b><br>
    - برای بیت‌کوین: <b>BTC-USD</b><br>
    - برای اپل: <b>AAPL</b><br>
    - برای اتریوم: <b>ETH-USD</b><br>
    - برای شاخص S&P500: <b>^GSPC</b><br>
    <br>
    <b>توضیحات بیشتر:</b><br>
    - نماد هر دارایی را می‌توانید در سایت <a href="https://finance.yahoo.com" target="_blank">Yahoo Finance</a> جستجو کنید.<br>
    - اگر چند نماد وارد می‌کنید، فقط با کاما جدا کنید و فاصله نگذارید.<br>
    - داده‌های دانلودشده دقیقاً مانند فایل CSV در ابزار استفاده می‌شوند.<br>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True, key="uploader"
)

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, 0.25, 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

with st.sidebar.expander("📥 دانلود داده آنلاین از یاهو فاینانس"):
    st.markdown("""
    <div dir="rtl" style="text-align: right; font-size: 14px">
    <b>راهنما:</b><br>
    - نماد هر دارایی را مطابق سایت یاهو فاینانس وارد کنید.<br>
    - نمادها را با کاما و بدون فاصله وارد کنید.<br>
    - مثال: <b>BTC-USD,AAPL,GOOGL,ETH-USD</b><br>
    - برای بیت‌کوین: <b>BTC-USD</b> <br>
    - برای اپل: <b>AAPL</b> <br>
    - برای اتریوم: <b>ETH-USD</b> <br>
    - برای شاخص S&P500: <b>^GSPC</b> <br>
    </div>
    """, unsafe_allow_html=True)
    tickers_input = st.text_input("نماد دارایی‌ها (با کاما و بدون فاصله)")
    start = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
    end = st.date_input("تاریخ پایان", value=pd.to_datetime("today"))
    download_btn = st.button("دریافت داده آنلاین")

downloaded_dfs = []
if download_btn and tickers_input.strip():
    tickers = [t.strip() for t in tickers_input.strip().split(",") if t.strip()]
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
        if data.empty:
            st.error("داده‌ای دریافت نشد!")
        else:
            for t in tickers:
                df, err = get_price_dataframe_from_yf(data, t)
                if df is not None:
                    df['Date'] = pd.to_datetime(df['Date'])
                    downloaded_dfs.append((t, df))
                    st.success(f"داده {t} با موفقیت دانلود شد.")
                    st.markdown(download_link(df, f"{t}_historical.csv"), unsafe_allow_html=True)
                else:
                    st.error(f"{err}")
    except Exception as ex:
        st.error(f"خطا در دریافت داده: {ex}")

if downloaded_dfs:
    st.markdown('<div dir="rtl" style="text-align: right;"><b>داده‌های دانلودشده از یاهو فاینانس:</b></div>', unsafe_allow_html=True)
    for t, df in downloaded_dfs:
        st.markdown(f"<div dir='rtl' style='text-align: right;'><b>{t}</b></div>", unsafe_allow_html=True)
        st.dataframe(df.head())

# (بقیه کد ابزار پرتفو بدون تغییر نسبت به نسخه قبل)

# ... ابزار پرتفو ...
# (همانند نسخه قبلی که داری، فقط کافیست تابع read_csv_file و get_price_dataframe_from_yf را جایگزین کنی)
