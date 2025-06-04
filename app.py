import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import re

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو، CVaR و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو، CVaR، مدل‌های کلاسیک و هوش مصنوعی")

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
        lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith('#')]
        if not lines:
            return None
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
            price_val = price_val.replace(' ', '').replace(',', '')
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

    adjusted_cov = cov_matrix.copy()
    preference_weights = []

    for i, name in enumerate(asset_names):
        if name in insured_assets:
            risk_scale = 1 - insured_assets[name]['loss_percent'] / 100
            adjusted_cov.iloc[i, :] *= risk_scale
            adjusted_cov.iloc[:, i] *= risk_scale
            preference_weights.append(1 / (std_devs[i] * risk_scale**0.7))  # وزن بیشتر در بیمه
        else:
            preference_weights.append(1 / std_devs[i])
    preference_weights = np.array(preference_weights)
    preference_weights /= np.sum(preference_weights)

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

    best_idx = np.argmin(np.abs(results[1] - user_risk))
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_sortino = results[3, best_idx]
    best_weights = results[5:, best_idx]

    best_cvar_idx = np.argmin(results[4])
    best_cvar_return = results[0, best_cvar_idx]
    best_cvar_risk = results[1, best_cvar_idx]
    best_cvar_cvar = results[4, best_cvar_idx]
    best_cvar_weights = results[5:, best_cvar_idx]

    # ============ مدل کلاسیک و رگرسیونی/هوش مصنوعی ============
    with st.expander("🤖 مدل‌های کلاسیک و یادگیری ماشین/هوش مصنوعی"):
        st.markdown("""
        <div dir="rtl" style="text-align: right">
        <b>این بخش پرتفو را با مدل‌های کلاسیک و هوش مصنوعی تحلیل می‌کند:</b><br>
        <ul>
        <li><b>حداقل واریانس جهانی (GMV):</b> کم‌ریسک‌ترین ترکیب ممکن پرتفو، صرفاً بر اساس کاهش نوسان.</li>
        <li><b>ماکزیمم نرخ شارپ (MSR):</b> بهترین ترکیب بازده نسبت به ریسک.</li>
        <li><b>شبیه‌سازی رگرسیونی:</b> پیش‌بینی قیمت آینده هر دارایی با ساده‌ترین مدل ML (رگرسیون خطی).</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # GMV
        gmv_weights = get_gmv_weights(cov_matrix)
        gmv_ret = np.dot(gmv_weights, mean_returns)
        gmv_risk = np.sqrt(np.dot(gmv_weights.T, np.dot(cov_matrix, gmv_weights)))
        st.markdown("#### 🟦 حداقل واریانس جهانی (GMV)")
        st.markdown(f'''
        <div dir="rtl" style="text-align: right">
        در این مدل، وزن دارایی‌ها طوری انتخاب می‌شود که واریانس (نوسان) پرتفو حداقل شود. مناسب برای افراد ریسک‌گریز.<br>
        <b>بازده سالانه پرتفو:</b> {gmv_ret:.2%}<br>
        <b>ریسک سالانه پرتفو:</b> {gmv_risk:.2%}<br>
        {'<br>'.join([f"🔹 <b>{asset_names[i]}</b>: {w*100:.2f}%" for i, w in enumerate(gmv_weights)])}
        </div>
        ''', unsafe_allow_html=True)

        fig_gmv = go.Figure(data=[go.Pie(labels=asset_names, values=gmv_weights*100, hole=0.5)])
        fig_gmv.update_layout(title="ترکیب وزنی پرتفو GMV")
        st.plotly_chart(fig_gmv, use_container_width=True)

        # Max Sharpe
        ms_weights = get_max_sharpe_weights(mean_returns, cov_matrix)
        ms_ret = np.dot(ms_weights, mean_returns)
        ms_risk = np.sqrt(np.dot(ms_weights.T, np.dot(cov_matrix, ms_weights)))
        ms_sharpe = (ms_ret - rf) / ms_risk
        st.markdown("#### 🟧 پرتفو با بالاترین نرخ شارپ (MSR)")
        st.markdown(f'''
        <div dir="rtl" style="text-align: right">
        این مدل پرتفو با بیشترین نسبت بازده به ریسک (Sharpe) را پیدا می‌کند. مناسب برای افرادی که به دنبال بیشترین بهره‌وری ریسک هستند.<br>
        <b>بازده سالانه پرتفو:</b> {ms_ret:.2%}<br>
        <b>ریسک سالانه پرتفو:</b> {ms_risk:.2%}<br>
        <b>نسبت شارپ:</b> {ms_sharpe:.2f}<br>
        {'<br>'.join([f"🔸 <b>{asset_names[i]}</b>: {w*100:.2f}%" for i, w in enumerate(ms_weights)])}
        </div>
        ''', unsafe_allow_html=True)

        fig_ms = go.Figure(data=[go.Pie(labels=asset_names, values=ms_weights*100, hole=0.5)])
        fig_ms.update_layout(title="ترکیب وزنی پرتفو بیشترین شارپ")
        st.plotly_chart(fig_ms, use_container_width=True)

        # رگرسیون خطی
        st.markdown("#### 🟩 شبیه‌سازی رگرسیونی بازده آینده هر دارایی (یادگیری ماشین)")
        st.markdown('''
        <div dir="rtl" style="text-align: right">
        با مدل رگرسیون خطی (ساده‌ترین مدل ML)، روند قیمت هر دارایی در آینده نزدیک پیش‌بینی می‌شود (مثلاً ماه آینده). هرچند این پیش‌بینی‌ها قطعیت ندارند، اما برای تخمین روند و سیگنال اولیه مفید است.
        </div>''', unsafe_allow_html=True)
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
        st.markdown('''
        <div dir="rtl" style="text-align: right;">
        این جدول پیش‌بینی قیمت رگرسیونی را برای هر دارایی نمایش می‌دهد. ستون "تغییر پیش‌بینی" میزان رشد یا کاهش احتمالی را برآورد می‌کند.
        </div>
        ''', unsafe_allow_html=True)

    # === ادامه کد اصلی (مونت‌کارلو، CVaR، Married Put، پیش‌بینی تصادفی، ...)

    # ... (همان کد خروجی قبلی تو از اینجا اجرا می‌شود - هیچ تغییری در فیچرها نداده‌ام)

    # ----- بقیه کد ابزار تو اینجا باقی بماند -----
    # (از st.subheader("📊 داشبورد خلاصه پرتفو") تا انتها هیچ تغییری نده، فقط این expander جدید اضافه شد)
else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
