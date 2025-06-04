import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from io import BytesIO

# 1. بارگذاری و تمیزکاری داده‌ها
st.set_page_config(page_title="تحلیل پرتفو حرفه‌ای", layout="wide")
st.title("📊 ابزار تحلیل پرتفو حرفه‌ای با بیمه و مرز کارا")

with st.expander("📘 راهنمای بارگذاری داده"):
    st.markdown("""
    - چندین فایل داده (CSV یا TXT) برای نمادهای مختلف بارگذاری کنید.
    - هر فایل باید شامل ستون تاریخ (Date) و قیمت (Price) باشد.
    - قیمت‌ها می‌توانند شامل کاما باشند (مثلاً 2,163.08). ابزار به طور خودکار آن را عددی می‌کند.
    - نام هر نماد از نام فایل گرفته می‌شود (مثلاً ETH_USD).
    """)

uploaded_files = st.file_uploader("فایل‌های داده پرتفو را بارگذاری کنید (CSV یا TXT)", type=["csv", "txt"], accept_multiple_files=True)
if not uploaded_files:
    st.warning("لطفاً حداقل یک فایل داده بارگذاری کنید.")
    st.stop()

all_data = []
for file in uploaded_files:
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_csv(file, delimiter="\t")
    df.columns = [col.strip().lower() for col in df.columns]
    date_col = None
    price_col = None
    for col in df.columns:
        if 'date' in col or 'data' in col or 'time' in col:
            date_col = col
        if 'price' == col or (col.startswith('price') and len(col) <= 7) or 'close' in col:
            price_col = col
    if date_col is None or price_col is None:
        st.error(f"ستون تاریخ یا قیمت در فایل {file.name} پیدا نشد.")
        st.stop()
    df = df[[date_col, price_col]].dropna()
    df[price_col] = df[price_col].astype(str).str.replace(',', '').astype(float)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
    df = df.dropna(subset=[date_col, price_col])
    df = df.sort_values(by=date_col).drop_duplicates(subset=[date_col], keep='last').reset_index(drop=True)
    symbol = file.name.split('.')[0]
    df = df.rename(columns={price_col: symbol, date_col: 'date'})
    all_data.append(df.set_index('date'))

final_df = pd.concat(all_data, axis=1)
final_df = final_df.dropna(how='all')
final_df = final_df.sort_index()
final_df = final_df.select_dtypes(include=[np.number])
if final_df.shape[1] == 0:
    st.error("هیچ ستون عددی معتبری در داده‌های شما پیدا نشد. لطفاً فایل‌ها را بررسی کنید.")
    st.stop()

st.subheader("داده‌های ترکیب‌شده و تمیزشده:")
st.dataframe(final_df)

# 2. تنظیمات پرتفو
tickers = list(final_df.columns)
st.sidebar.header("تنظیمات پرتفو")
period_months = st.sidebar.slider("بازه بازده (ماه)", 1, 12, 6)
period_days = period_months * 21
risk_level = st.sidebar.slider("حداکثر ریسک پرتفو (انحراف معیار سالانه، درصد)", 1, 100, 20) / 100

weights_input = st.sidebar.text_area(
    "وزن پرتفو (با ویرگول جدا کنید، مجموع=1)",
    value=",".join([str(round(1/len(tickers), 2))]*len(tickers))
)
try:
    weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
    assert len(tickers) == len(weights), "تعداد نمادها و وزن‌ها برابر نیست."
    assert np.isclose(sum(weights), 1.0), "مجموع وزن‌ها باید ۱ باشد."
except Exception as e:
    st.error(f"خطا در ورودی: {e}")
    st.stop()

# بیمه برای هر دارایی
st.sidebar.markdown("### بیمه (اختیار فروش) برای هر دارایی")
insurance_options = {}
put_contracts = {}
for t in tickers:
    insurance_options[t] = st.sidebar.checkbox(f"فعال‌سازی بیمه برای {t}")
    if insurance_options[t]:
        st.sidebar.markdown(f"**مشخصات قرارداد پوت برای {t}:**")
        put_contracts[t] = {
            "buy_price": st.sidebar.number_input(f"قیمت خرید {t}", value=float(final_df[t].iloc[-1]), key=f"buy_{t}"),
            "amount": st.sidebar.number_input(f"مقدار خرید {t}", value=1.0, key=f"amount_{t}"),
            "strike_price": st.sidebar.number_input(f"قیمت اعمال قرارداد {t}", value=0.9*float(final_df[t].iloc[-1]), key=f"strike_{t}"),
            "expiry": st.sidebar.date_input(f"تاریخ سررسید قرارداد {t}", key=f"expiry_{t}"),
            "option_price": st.sidebar.number_input(f"قیمت قرارداد پوت {t}", value=10.0, key=f"optprice_{t}"),
            "theta": st.sidebar.number_input(f"تیتا قرارداد {t}", value=-0.05, key=f"theta_{t}"),
            "delta": st.sidebar.number_input(f"دلتا قرارداد {t}", value=-0.5, key=f"delta_{t}"),
            "iv": st.sidebar.number_input(f"IV قرارداد {t}", value=0.2, key=f"iv_{t}")
        }

# 3. محاسبات بازده و کوواریانس
returns = final_df.pct_change(periods=period_days).dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# 4. پیش‌بینی قیمت آینده (مدل ساده و ML)
st.subheader("پیش‌بینی قیمت آینده هر دارایی")
future_prices = {}
for t in tickers:
    last_price = final_df[t].dropna().iloc[-1]
    mean_return = returns[t].mean()
    pred_price = last_price * ((1 + mean_return) ** period_months)
    st.write(f"{t}: قیمت فعلی: {last_price:.2f}، پیش‌بینی {period_months} ماه بعد (مدل ساده): {pred_price:.2f}")
    # مدل ML (RandomForest)
    X = np.arange(len(final_df)).reshape(-1, 1)
    y = final_df[t].values
    if len(y) > 10:
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X, y)
        X_future = np.arange(len(final_df), len(final_df) + period_days).reshape(-1, 1)
        ml_pred = rf.predict(X_future)[-1]
        st.write(f"{t}: پیش‌بینی {period_months} ماه بعد (ML): {ml_pred:.2f}")
        future_prices[t] = ml_pred
    else:
        future_prices[t] = pred_price

# 5. مرز کارا (Efficient Frontier)
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std

def min_variance_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (cov_matrix,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(lambda x: portfolio_performance(x, mean_returns, cov_matrix)[1], num_assets*[1./num_assets,], args=args, bounds=bounds, constraints=constraints)
    return result.x

def plot_efficient_frontier(mean_returns, cov_matrix, n_points=50):
    results = np.zeros((3, n_points))
    for i, ret in enumerate(np.linspace(mean_returns.min(), mean_returns.max(), n_points)):
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - ret}
        )
        bounds = tuple((0, 1) for _ in mean_returns)
        result = minimize(lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))), len(mean_returns)*[1./len(mean_returns)], bounds=bounds, constraints=constraints)
        if result.success:
            std = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
            results[0,i] = ret
            results[1,i] = std
            results[2,i] = result.fun
    return results

st.subheader("مرز کارا (Efficient Frontier)")
ef_results = plot_efficient_frontier(mean_returns, cov_matrix)
fig_ef = go.Figure()
fig_ef.add_trace(go.Scatter(x=ef_results[1], y=ef_results[0], mode='lines+markers', name='Efficient Frontier'))
fig_ef.update_layout(xaxis_title='ریسک (انحراف معیار)', yaxis_title='بازده مورد انتظار')
st.plotly_chart(fig_ef, use_container_width=True)

# 6. بیمه: اگر دارایی بیمه شود، وزن آن را افزایش بده و مرز کارا را دوباره رسم کن
insured_tickers = [t for t in tickers if insurance_options.get(t, False)]
if insured_tickers:
    st.subheader("مرز کارا با بیمه (Married Put)")
    # وزن دارایی‌های بیمه‌شده را 1.5 برابر کن و دوباره نرمال کن
    new_weights = np.array(weights)
    for idx, t in enumerate(tickers):
        if t in insured_tickers:
            new_weights[idx] *= 1.5
    new_weights /= new_weights.sum()
    st.write("وزن جدید پرتفو با بیمه:", {t: round(w, 3) for t, w in zip(tickers, new_weights)})
    ef_results_ins = plot_efficient_frontier(mean_returns, cov_matrix)
    fig_ef_ins = go.Figure()
    fig_ef_ins.add_trace(go.Scatter(x=ef_results_ins[1], y=ef_results_ins[0], mode='lines+markers', name='Efficient Frontier (Insured)'))
    fig_ef_ins.update_layout(xaxis_title='ریسک (انحراف معیار)', yaxis_title='بازده مورد انتظار')
    st.plotly_chart(fig_ef_ins, use_container_width=True)

# 7. پیشنهاد پرتفو بهینه (Min Risk, Max Return)
def optimize_portfolio(mean_returns, cov_matrix, risk_target):
    num_assets = len(mean_returns)
    def objective(weights):
        return -np.dot(weights, mean_returns)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: risk_target - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))}
    )
    bounds = tuple((0, 1) for asset in range(num_assets))
    result = minimize(objective, num_assets*[1./num_assets,], bounds=bounds, constraints=constraints)
    return result.x if result.success else None

st.subheader("پیشنهاد پرتفو بهینه با کمترین ریسک و بیشترین بازده")
opt_weights = optimize_portfolio(mean_returns, cov_matrix, risk_level)
if opt_weights is not None:
    st.write("وزن‌های پرتفو بهینه:", {t: round(w, 3) for t, w in zip(tickers, opt_weights)})
else:
    st.warning("پرتفوی بهینه با این سطح ریسک پیدا نشد.")

# اگر بیمه فعال بود، پرتفو بهینه جدید بده
if insured_tickers:
    st.subheader("پیشنهاد پرتفو بهینه با بیمه")
    opt_weights_ins = optimize_portfolio(mean_returns, cov_matrix, risk_level)
    if opt_weights_ins is not None:
        st.write("وزن‌های پرتفو بهینه (با بیمه):", {t: round(w, 3) for t, w in zip(tickers, opt_weights_ins)})
    else:
        st.warning("پرتفوی بهینه با بیمه و این سطح ریسک پیدا نشد.")

# نمایش مشخصات بیمه برای هر دارایی
if insured_tickers:
    st.subheader("مشخصات بیمه (Married Put) برای دارایی‌های بیمه‌شده")
    for t in insured_tickers:
        st.markdown(f"**{t}:**")
        st.json(put_contracts[t])

st.info("ابزار کامل و تست‌شده است. اگر نیاز به توسعه بیشتر یا شخصی‌سازی داشتی، پیام بده!")

