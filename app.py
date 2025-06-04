import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import re
import io

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو، CVaR، مدل‌های کلاسیک و یادگیری ماشین", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو، CVaR، مدل‌های کلاسیک و یادگیری ماشین")

# ===== راهنمای یاهو فاینانس =====
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

# =============== تابع خواندن مقاوم فایل ===============
def smart_read_file(file):
    try:
        content = file.read()
        try:
            content = content.decode("utf-8")
        except Exception:
            content = content.decode("latin1")
        seps = [',',';','|','\t']
        sep_counts = [(s, content.count(s)) for s in seps]
        sep = max(sep_counts, key=lambda x:x[1])[0] if max(sep_counts, key=lambda x:x[1])[1] > 0 else ','
        df = pd.read_csv(io.StringIO(content), sep=sep)
        # پیدا کردن ستون تاریخ و قیمت (فارسی یا انگلیسی)
        col_date = [col for col in df.columns if 'date' in col.lower() or 'تاریخ' in col.lower()]
        col_price = [col for col in df.columns if 'price' in col.lower() or 'قیمت' in col.lower()]
        if not col_date or not col_price:
            st.error("ستون تاریخ یا قیمت یافت نشد. نام ستون‌ها باید شامل 'Date' یا 'تاریخ' و 'Price' یا 'قیمت' باشد.")
            return None
        df = df[[col_date[0], col_price[0]]].copy()
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # پاکسازی قیمت: حذف فاصله، ویرگول فارسی و انگلیسی و نقطه یا هر کاراکتر غیرعددی (به جز نقطه و -)
        df['Price'] = (
            df['Price'].astype(str)
              .str.replace('٬', '', regex=False)  # ویرگول فارسی
              .str.replace(',', '', regex=False)  # ویرگول انگلیسی
              .str.replace(' ', '', regex=False)
              .str.replace(r'[^\d\.\-]', '', regex=True)  # فقط اعداد و نقطه و -
        )
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df.sort_values('Date')
        if len(df) < 3:
            return None
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل: {e}")
        return None

# =============== مدل‌های کلاسیک و رگرسیونی ===============
def get_gmv_weights(cov_matrix):
    n = cov_matrix.shape[0]
    args = (cov_matrix,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(lambda w, cov: w.T @ cov @ w, n*[1./n], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def get_max_sharpe_weights(mean_returns, cov_matrix, rf=0):
    n = len(mean_returns)
    def neg_sharpe(w, mean, cov, rf):
        port_return = np.dot(w, mean)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(port_return - rf) / port_vol
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
    bounds = tuple((0, 1) for _ in range(n))
    result = minimize(neg_sharpe, n*[1./n], args=(mean_returns, cov_matrix, rf), method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def regression_forecast(prices, periods_ahead=1):
    prices = prices.dropna()
    if len(prices) < 10:
        return np.nan
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices.values
    model = LinearRegression().fit(X, y)
    pred = model.predict([[len(prices) + periods_ahead - 1]])
    return float(pred)

# =============== سایدبار ===============
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV یا TXT)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV یا TXT آپلود کنید (هر دارایی یک فایل)", type=['csv', 'txt'], accept_multiple_files=True, key="uploader"
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

if uploaded_files or downloaded_dfs:
    prices_df = pd.DataFrame()
    asset_names = []
    insured_assets = {}

    # داده‌های دانلودشده
    for t, df in downloaded_dfs:
        name = t
        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"داده آنلاین {name} باید دارای ستون‌های 'Date' و 'Price' باشد.")
            continue
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

    # فایل‌های آپلودی کاربر (هوشمند)
    for file in uploaded_files:
        df = smart_read_file(file)
        if df is None:
            st.warning(f"فایل {file.name} باید حداقل سه سطر داده سالم و ستون‌های 'Date' و 'Price' داشته باشد.")
            continue
        name = file.name.split('.')[0]
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

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))

    # ============ مدل‌های کلاسیک و ML/R ==================
    with st.expander("🤖 مدل‌های کلاسیک و یادگیری ماشین (GMV, Sharpe, رگرسیون)"):
        st.markdown("""
        <div dir="rtl" style="text-align: right">
        در این بخش، سه روش استاندارد و هوشمند برای بهینه‌سازی پرتفو و پیش‌بینی آینده دارایی‌ها ارائه شده است:
        <ol>
          <li><b>حداقل واریانس جهانی (GMV):</b> کم‌ریسک‌ترین پرتفو ممکن بدون توجه به بازده.</li>
          <li><b>ماکزیمم نرخ شارپ (Maximum Sharpe Ratio):</b> پیدا کردن پرتفو با بهترین نسبت بازده به ریسک.</li>
          <li><b>پیش‌بینی رگرسیونی قیمت دارایی‌ها (یادگیری ماشین):</b> استفاده از رگرسیون خطی برای پیش‌بینی قیمت آینده هر دارایی.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        # ====== محاسبه و نمایش GMV ======
        gmv_weights = get_gmv_weights(cov_matrix)
        gmv_ret = np.dot(gmv_weights, mean_returns)
        gmv_risk = np.sqrt(np.dot(gmv_weights.T, np.dot(cov_matrix, gmv_weights)))
        st.markdown("#### 📘 حداقل واریانس جهانی (Global Minimum Variance)")
        st.markdown(f'''
        <div dir="rtl" style="text-align: right">
        این مدل کم‌ریسک‌ترین ترکیب پرتفو را صرف‌نظر از بازده انتخاب می‌کند.<br>
        <b>بازده سالانه پرتفو:</b> {gmv_ret:.2%}<br>
        <b>ریسک سالانه پرتفو:</b> {gmv_risk:.2%}<br>
        {'<br>'.join([f"🔹 <b>{asset_names[i]}</b>: {w*100:.2f}%" for i, w in enumerate(gmv_weights)])}
        </div>
        ''', unsafe_allow_html=True)
        fig_gmv = go.Figure(data=[go.Pie(labels=asset_names, values=gmv_weights*100, hole=0.5)])
        fig_gmv.update_layout(title="ترکیب وزنی پرتفو GMV")
        st.plotly_chart(fig_gmv, use_container_width=True)

        # ====== محاسبه و نمایش Max Sharpe ======
        ms_weights = get_max_sharpe_weights(mean_returns, cov_matrix)
        ms_ret = np.dot(ms_weights, mean_returns)
        ms_risk = np.sqrt(np.dot(ms_weights.T, np.dot(cov_matrix, ms_weights)))
        ms_sharpe = (ms_ret) / ms_risk
        st.markdown("#### 📙 پرتفو با بالاترین نرخ شارپ (Maximum Sharpe Ratio)")
        st.markdown(f'''
        <div dir="rtl" style="text-align: right">
        این مدل پرتفو با بهترین نسبت بازده به ریسک را می‌سازد.<br>
        <b>بازده سالانه پرتفو:</b> {ms_ret:.2%}<br>
        <b>ریسک سالانه پرتفو:</b> {ms_risk:.2%}<br>
        <b>نسبت شارپ:</b> {ms_sharpe:.2f}<br>
        {'<br>'.join([f"🔸 <b>{asset_names[i]}</b>: {w*100:.2f}%" for i, w in enumerate(ms_weights)])}
        </div>
        ''', unsafe_allow_html=True)
        fig_ms = go.Figure(data=[go.Pie(labels=asset_names, values=ms_weights*100, hole=0.5)])
        fig_ms.update_layout(title="ترکیب وزنی پرتفو Sharpe")
        st.plotly_chart(fig_ms, use_container_width=True)

        # ====== پیش‌بینی رگرسیونی ======
        st.markdown("#### 📗 پیش‌بینی قیمت آینده با رگرسیون (یادگیری ماشین)")
        reg_rows = []
        for name in asset_names:
            last_price = resampled_prices[name].dropna()
            reg_pred = regression_forecast(last_price, periods_ahead=1)
            delta = reg_pred - last_price.iloc[-1] if not np.isnan(reg_pred) else np.nan
            reg_rows.append({
                "دارایی": name,
                "قیمت فعلی": last_price.iloc[-1],
                "پیش‌بینی رگرسیونی ماه بعد": reg_pred,
                "تغییر پیش‌بینی": delta
            })
        reg_df = pd.DataFrame(reg_rows)
        st.dataframe(reg_df.set_index("دارایی"), use_container_width=True)
        st.markdown("""
        <div dir="rtl" style="text-align: right">
        این جدول با رگرسیون خطی (مدل ML ساده)، قیمت احتمالی هر دارایی در ماه آینده را تخمین می‌زند.<br>
        توجه: این مدل فقط روند خطی را می‌سنجد و ممکن است برای بازارهای پرتلاطم مناسب نباشد.
        </div>
        """, unsafe_allow_html=True)

    # ========== ادامه ابزار (مونت‌کارلو، CVaR، Married Put، پیش‌بینی تصادفی و ...) ==========
    # ... (همان کد کامل قبلی، مطابق نسخه‌های قبلی این چت) ...

else:
    st.warning("⚠️ لطفاً فایل‌های CSV یا TXT شامل ستون‌های Date و Price را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
