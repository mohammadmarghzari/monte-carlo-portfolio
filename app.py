import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import io
import re

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو، CVaR و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو، CVaR و استراتژی Married Put")

with st.expander("📄 راهنمای فایل ورودی (csv یا txt)"):
    st.markdown("""
    <div dir="rtl" style="text-align: right;">
    <b>برای عملکرد صحیح ابزار:</b> فقط دو ستون <b>Date</b> و <b>Price</b> کافی است.<br>
    سایر ستون‌ها (Open, High, Low, Volume, ...) حذف می‌شوند.<br>
    ابزار به طور خودکار فایل را تمیز می‌کند؛ اما اگر به خطا خوردید، مطمئن شوید فایل شما مانند نمونه زیر باشد:
    </div>
    """, unsafe_allow_html=True)
    st.code(
"""Date,Price
2025-06-02,13.86
2025-06-01,14.06
""", language="text")
    st.info("اگر فایل شما ستون اضافی داشت یا عددها در کوتیشن بودند، نگران نباشید! ابزار به‌طور خودکار تمیز می‌کند.")

sample_txt = """# نمونه فایل txt برای ابزار تحلیل پرتفو
Date,Price
2025-06-02,13.86
2025-06-01,14.06
"""
st.download_button("دانلود نمونه فایل txt", sample_txt, file_name="sample.txt", mime="text/plain")

def smart_read_file(file):
    """
    خواندن و تمیزکاری هوشمند فایل csv/txt (هر جداکننده و هر ساختاری!)
    فقط ستون تاریخ و قیمت را برمی‌گرداند.
    """
    try:
        content = file.read()
        try:
            content = content.decode("utf-8")
        except Exception:
            content = content.decode("latin1")

        # تشخیص جداکننده محتمل
        seps = [',',';','|','\t']
        sep_counts = [(s, content.count(s)) for s in seps]
        sep = max(sep_counts, key=lambda x:x[1])[0] if max(sep_counts, key=lambda x:x[1])[1] > 0 else ','

        # فقط خطوط غیرخالی و غیرتوضیحی
        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith('#')]
        if not lines:
            return None

        # جداسازی هدر و داده‌ها
        header = [x.strip().replace('"','').replace("'",'') for x in re.split(sep, lines[0])]
        def find_col(colnames, candidates):
            for c in candidates:
                for i, h in enumerate(colnames):
                    if c.lower() in h.lower():
                        return i
            return None

        date_idx = find_col(header, ['date', 'تاریخ'])
        price_idx = find_col(header, ['price', 'قیمت'])
        if date_idx is None or price_idx is None:
            return None

        data_rows = []
        for row in lines[1:]:
            parts = [x.strip().replace('"','').replace("'",'') for x in re.split(sep, row)]
            if len(parts) <= max(date_idx, price_idx): continue
            date_val = parts[date_idx]
            price_val = parts[price_idx]
            price_val = price_val.replace(' ', '').replace(',', '.')
            price_val = re.sub(r'[^\d\.\-]', '', price_val)
            try:
                price_float = float(price_val)
            except:
                continue
            data_rows.append([date_val, price_float])

        if not data_rows:
            return None

        df = pd.DataFrame(data_rows, columns=['Date', 'Price'])
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df.sort_values('Date')
        if len(df) < 3:
            return None
        return df
    except Exception as e:
        return None

st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV یا TXT)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV یا TXT آپلود کنید (هر دارایی یک فایل)",
    type=['csv', 'txt'],
    accept_multiple_files=True,
    key="uploader"
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
    - مثال: <b>BTC-USD,AAPL,GOOGL,ETH-USD</b>
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
        if not data.empty:
            for t in tickers:
                if len(tickers) == 1:
                    df = data.reset_index()[['Date', 'Close']].rename(columns={'Close': 'Price'})
                else:
                    if t in data.columns.levels[0]:
                        df_t = data[t].reset_index()[['Date', 'Close']].rename(columns={'Close': 'Price'})
                        df = df_t
                    else:
                        st.warning(f"داده‌ای برای نماد {t} دریافت نشد.")
                        continue
                df['Date'] = pd.to_datetime(df['Date'])
                downloaded_dfs.append((t, df))
                st.success(f"داده {t} با موفقیت دانلود شد.")
        else:
            st.error("داده‌ای دریافت نشد!")
    except Exception as ex:
        st.error(f"خطا در دریافت داده: {ex}")

if downloaded_dfs:
    st.markdown('<div dir="rtl" style="text-align: right;"><b>داده‌های دانلودشده از یاهو فاینانس:</b></div>', unsafe_allow_html=True)
    for t, df in downloaded_dfs:
        st.markdown(f"<div dir='rtl' style='text-align: right;'><b>{t}</b></div>", unsafe_allow_html=True)
        st.dataframe(df.head())

def get_unique_name(existing_names, base_name):
    if base_name not in existing_names:
        return base_name
    i = 2
    while f"{base_name}_{i}" in existing_names:
        i += 1
    return f"{base_name}_{i}"

if uploaded_files or downloaded_dfs:
    prices_df = pd.DataFrame()
    asset_names = []
    insured_assets = {}
    existing_names = set()

    # داده‌های دانلودشده
    for t, df in downloaded_dfs:
        name = get_unique_name(existing_names, t)
        existing_names.add(name)
        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"داده آنلاین {name} باید دارای ستون‌های 'Date' و 'Price' باشد.")
            continue
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

    # فایل‌های آپلودی کاربر
    for file in uploaded_files:
        base_name = file.name.split('.')[0]
        name = get_unique_name(existing_names, base_name)
        existing_names.add(name)
        df = smart_read_file(file)
        if df is None or 'Date' not in df.columns or 'Price' not in df.columns or len(df) < 3:
            st.warning(f"فایل {name} باید دارای حداقل سه سطر داده و ستون‌های 'Date' و 'Price' باشد.")
            continue
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

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

    if prices_df.empty or prices_df.shape[1] < 2:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد یا حداقل دو دارایی معتبر ندارید.")
        st.stop()

    st.subheader("🧪 پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail())
    prices_df = prices_df.dropna(axis=1, how='any')
    if prices_df.shape[1] < 2:
        st.error("حداقل دو دارایی معتبر نیاز است (هر دارایی ستون جدا).")
        st.stop()

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))

    adjusted_cov = cov_matrix.copy()
    preference_weights = []

    for i, name in enumerate(asset_names):
        if name in insured_assets:
            risk_scale = 1 - insured_assets[name]['loss_percent'] / 100
            adjusted_cov.iloc[i, :] *= risk_scale
            adjusted_cov.iloc[:, i] *= risk_scale
            preference_weights.append(1 / (std_devs[i] * risk_scale**0.7))
        else:
            preference_weights.append(1 / std_devs[i])
    preference_weights = np.array(preference_weights)
    preference_weights /= np.sum(preference_weights)

    # بررسی سلامت کوواریانس
    if np.any(np.isnan(adjusted_cov)) or np.any(np.isinf(adjusted_cov)):
        st.error("ماتریس کوواریانس ناسالم است (شامل NaN یا Inf). داده‌ها را بررسی کنید (مثلاً داده کم یا فایل خراب).")
        st.stop()
    if not np.allclose(adjusted_cov, adjusted_cov.T):
        st.error("ماتریس کوواریانس متقارن نیست.")
        st.stop()
    eigvals = np.linalg.eigvals(adjusted_cov)
    if np.any(eigvals <= 0):
        st.error("ماتریس کوواریانس مثبت معین نیست یا داده‌ها کافی نیست یا دارایی‌های تکراری دارید.")
        st.stop()

    # ... ادامه کد ابزار مثل نمونه‌های قبلی (شبیه‌سازی مونت‌کارلو و داشبورد)

else:
    st.warning("⚠️ لطفاً فایل‌های CSV یا TXT معتبر شامل ستون‌های Date و Price را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
