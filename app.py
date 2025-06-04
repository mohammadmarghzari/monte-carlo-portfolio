import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو و مدل‌های کلاسیک", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو و مدل‌های کلاسیک")

# تابع خواندن فایل مقاوم با تشخیص جداکننده (حتی تب) و استخراج صحیح قیمت
def smart_read_file(file):
    try:
        content = file.read()
        try:
            content = content.decode("utf-8")
        except Exception:
            content = content.decode("latin1")
        sample = "\n".join(content.splitlines()[:3])
        if "\t" in sample:
            sep = "\t"
        elif ";" in sample:
            sep = ";"
        elif "," in sample and sample.count(",") > sample.count("\t") + sample.count(";"):
            sep = ","
        elif "|" in sample:
            sep = "|"
        else:
            sep = "\t"  # پیش‌فرض برای داده‌های مالی
        df = pd.read_csv(io.StringIO(content), sep=sep)
        col_date = [col for col in df.columns if 'date' in col.lower() or 'تاریخ' in col.lower()]
        col_price = [col for col in df.columns if 'price' in col.lower() or 'قیمت' in col.lower()]
        if not col_date or not col_price:
            st.error("ستون تاریخ یا قیمت یافت نشد. نام ستون‌ها باید شامل 'Date' یا 'تاریخ' و 'Price' یا 'قیمت' باشد.")
            return None
        df = df[[col_date[0], col_price[0]]].copy()
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=False)
        df['Price'] = (
            df['Price'].astype(str)
                .str.replace('٬', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.replace(' ', '', regex=False)
                .str.replace(r'[^\d\.\-]', '', regex=True)
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

st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV/TXT - جداکننده هوشمند)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV یا TXT آپلود کنید (هر دارایی یک فایل جدا)", type=['csv', 'txt'], accept_multiple_files=True, key="uploader"
)

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'هفتگی'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'هفتگی': 'W'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'هفتگی': 52}[period]

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []

    for file in uploaded_files:
        df = smart_read_file(file)
        if df is None:
            st.warning(f"فایل {file.name} معتبر نیست یا کمتر از ۳ سطر دیتا دارد.")
            continue
        name = file.name.split('.')[0]
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    st.subheader("🧪 پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail())

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor

    with st.expander("📈 مدل‌های کلاسیک (GMV, Sharpe, رگرسیون)"):
        gmv_weights = get_gmv_weights(cov_matrix)
        gmv_ret = np.dot(gmv_weights, mean_returns)
        gmv_risk = np.sqrt(np.dot(gmv_weights.T, np.dot(cov_matrix, gmv_weights)))
        st.markdown("#### 📘 حداقل واریانس جهانی (GMV)")
        st.markdown(f'''
        <div dir="rtl" style="text-align: right">
        بازده سالانه: {gmv_ret:.2%} <br>
        ریسک سالانه: {gmv_risk:.2%} <br>
        {'<br>'.join([f"🔹 <b>{asset_names[i]}</b>: {w*100:.2f}%" for i, w in enumerate(gmv_weights)])}
        </div>
        ''', unsafe_allow_html=True)
        fig_gmv = go.Figure(data=[go.Pie(labels=asset_names, values=gmv_weights*100, hole=0.5)])
        fig_gmv.update_layout(title="ترکیب وزنی پرتفو GMV")
        st.plotly_chart(fig_gmv, use_container_width=True)

        ms_weights = get_max_sharpe_weights(mean_returns, cov_matrix)
        ms_ret = np.dot(ms_weights, mean_returns)
        ms_risk = np.sqrt(np.dot(ms_weights.T, np.dot(cov_matrix, ms_weights)))
        ms_sharpe = (ms_ret) / ms_risk
        st.markdown("#### 📙 پرتفو با بالاترین نرخ شارپ (Maximum Sharpe Ratio)")
        st.markdown(f'''
        <div dir="rtl" style="text-align: right">
        بازده سالانه: {ms_ret:.2%} <br>
        ریسک سالانه: {ms_risk:.2%} <br>
        نسبت شارپ: {ms_sharpe:.2f} <br>
        {'<br>'.join([f"🔸 <b>{asset_names[i]}</b>: {w*100:.2f}%" for i, w in enumerate(ms_weights)])}
        </div>
        ''', unsafe_allow_html=True)
        fig_ms = go.Figure(data=[go.Pie(labels=asset_names, values=ms_weights*100, hole=0.5)])
        fig_ms.update_layout(title="ترکیب وزنی پرتفو Sharpe")
        st.plotly_chart(fig_ms, use_container_width=True)

        st.markdown("#### 📗 پیش‌بینی قیمت آینده با رگرسیون (یادگیری ماشین)")
        reg_rows = []
        for name in asset_names:
            last_price = resampled_prices[name].dropna()
            reg_pred = regression_forecast(last_price, periods_ahead=1)
            delta = reg_pred - last_price.iloc[-1] if not np.isnan(reg_pred) else np.nan
            reg_rows.append({
                "دارایی": name,
                "قیمت فعلی": last_price.iloc[-1],
                "پیش‌بینی ماه بعد": reg_pred,
                "تغییر پیش‌بینی": delta
            })
        reg_df = pd.DataFrame(reg_rows)
        st.dataframe(reg_df.set_index("دارایی"), use_container_width=True)

else:
    st.warning("⚠️ لطفاً فایل‌های CSV یا TXT با ستون‌های Date و Price (یا تاریخ و قیمت) آپلود کنید (هر دارایی یک فایل جداگانه).")
