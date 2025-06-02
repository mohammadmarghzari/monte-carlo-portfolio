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
    ابزار به طور خودکار فایل شما را تمیز می‌کند؛ اما اگر به خطا خوردید، مطمئن شوید فایل شما مانند نمونه زیر باشد:
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

def clean_date(val):
    """تاریخ را به فرمت YYYY-MM-DD تبدیل می‌کند (در حد امکان)"""
    try:
        return pd.to_datetime(val, dayfirst=False, errors='coerce').strftime('%Y-%m-%d')
    except Exception:
        return val

def clean_price(val):
    """قیمت را به float تمیز تبدیل می‌کند"""
    if pd.isnull(val):
        return np.nan
    s = str(val)
    s = re.sub(r'["\',]', '', s)
    s = re.sub(r'[^\d\.\-]', '', s)
    try:
        return float(s)
    except Exception:
        return np.nan

def read_uploaded_file(file):
    try:
        # خواندن فایل به صورت خام (csv یا txt)
        content = file.read()
        try:
            # ابتدا بررسی کنیم فایل utf-8 است
            content = content.decode("utf-8")
        except Exception:
            # اگر نبود، با encoding دیگر (مثلاً latin1) بخوانیم
            content = content.decode("latin1")
        # حذف خطوط توضیحی
        lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith('#')]
        # تبدیل لیست به متن دوباره
        data_str = '\n'.join(lines)
        # تشخیص جداکننده: اگر ; بیشتر از , است، جداکننده را ; قرار بده
        delimiter = ',' if data_str.count(',') >= data_str.count(';') else ';'
        # خواندن دیتا
        df = pd.read_csv(io.StringIO(data_str), delimiter=delimiter)
        # تمیز کردن نام ستون‌ها (حذف کوتیشن و فاصله)
        df.columns = [c.strip().replace('"', '').replace("'", "").lower() for c in df.columns]
        # سعی کن ستون‌های مناسب پیدا کنی
        # پیدا کردن ستون تاریخ
        date_cols = [c for c in df.columns if 'date' in c]
        price_cols = [c for c in df.columns if 'price' in c]
        if not date_cols or not price_cols:
            st.error("فایل باید شامل ستون‌های Date و Price باشد (حتی اگر اسمشون مثلاً 'Close Price' باشه).")
            return None
        # انتخاب اولین ستون تاریخ و قیمت
        date_col = date_cols[0]
        price_col = price_cols[0]
        df = df[[date_col, price_col]]
        df.columns = ['Date', 'Price']
        # تمیز کردن تاریخ و قیمت
        df['Date'] = df['Date'].map(clean_date)
        df['Price'] = df['Price'].map(clean_price)
        # حذف سطرهایی که Price یا Date ندارند
        df = df.dropna(subset=['Date', 'Price'])
        # مرتب‌سازی بر اساس تاریخ
        df = df.sort_values('Date')
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
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
        df = read_uploaded_file(file)
        if df is None:
            continue
        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"فایل {name} باید دارای ستون‌های 'Date' و 'Price' باشد.")
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

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    st.subheader("🧪 پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail())
    # حذف دارایی‌هایی که داده معتبر ندارند یا تکراری‌اند
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
        st.error("ماتریس کوواریانس ناسالم است (شامل NaN یا Inf). داده‌ها را بررسی کنید.")
        st.stop()
    if not np.allclose(adjusted_cov, adjusted_cov.T):
        st.error("ماتریس کوواریانس متقارن نیست.")
        st.stop()
    eigvals = np.linalg.eigvals(adjusted_cov)
    if np.any(eigvals <= 0):
        st.error("ماتریس کوواریانس مثبت معین نیست یا داده‌ها کافی نیست یا دارایی‌های تکراری دارید.")
        st.stop()

    n_portfolios = 10000
    n_mc = 1000
    results = np.zeros((5 + len(asset_names), n_portfolios))
    np.random.seed(42)
    rf = 0

    downside = returns.copy()
    downside[downside > 0] = 0

    for i in range(n_portfolios):
        weights = np.random.random(len(asset_names)) * preference_weights
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))
        downside_risk = np.sqrt(np.dot(weights.T, np.dot(downside.cov() * annual_factor, weights)))
        sharpe_ratio = (port_return - rf) / port_std
        sortino_ratio = (port_return - rf) / downside_risk if downside_risk > 0 else np.nan

        mc_sims = np.random.multivariate_normal(mean_returns/annual_factor, adjusted_cov/annual_factor, n_mc)
        port_mc_returns = np.dot(mc_sims, weights)
        VaR = np.percentile(port_mc_returns, (1 - cvar_alpha) * 100)
        CVaR = port_mc_returns[port_mc_returns <= VaR].mean() if np.any(port_mc_returns <= VaR) else VaR

        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3, i] = sortino_ratio
        results[4, i] = -CVaR
        results[5:, i] = weights

    # ... ادامه کد ابزار مثل قبل است (داشبورد و نمودارها)
    # اگر لازم است بگو تا ادامه را هم اینجا بنویسم.
else:
    st.warning("⚠️ لطفاً فایل‌های CSV یا TXT معتبر شامل ستون‌های Date و Price را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
