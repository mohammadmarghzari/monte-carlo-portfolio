import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from io import BytesIO

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو، CVaR و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو، CVaR و استراتژی Married Put")

# --- راهنمای استفاده ---
with st.expander("📘 راهنمای دریافت داده آنلاین از یاهو فاینانس"):
    st.markdown("""
    - نماد سهام را به فرمت یاهو فاینانس وارد کنید (مثال: AAPL برای اپل، MSFT برای مایکروسافت).
    - برای بورس تهران نماد را به صورت `نماد.IR` وارد کنید (مثال: فولاد.IR).
    - وزن پرتفو باید بین ۰ و ۱ باشد و مجموع آن‌ها ۱ شود.
    - تعداد شبیه‌سازی و بازه زمانی را بر اساس نیاز تنظیم کنید.
    """)

# --- ورودی کاربر برای نمادها و وزن‌ها ---
st.sidebar.header("تنظیمات پرتفو")
tickers = st.sidebar.text_area("نمادها (با ویرگول جدا کنید)", value="AAPL,MSFT,GOOGL")
weights_input = st.sidebar.text_area("وزن پرتفو (با ویرگول جدا کنید، مجموع=1)", value="0.4,0.3,0.3")
try:
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
    assert len(tickers) == len(weights), "تعداد نمادها و وزن‌ها برابر نیست."
    assert np.isclose(sum(weights), 1.0), "مجموع وزن‌ها باید ۱ باشد."
except Exception as e:
    st.error(f"خطا در ورودی: {e}")
    st.stop()

start_date = st.sidebar.date_input("تاریخ شروع", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("تاریخ پایان", pd.to_datetime("today"))
n_sim = st.sidebar.number_input("تعداد شبیه‌سازی مونت‌کارلو", 100, 5000, 1000)
n_days = st.sidebar.number_input("تعداد روزهای پیش‌بینی", 30, 365, 180)
confidence_level = st.sidebar.slider("سطح اطمینان VaR/CVaR", 90, 99, 95)

initial_investment = st.sidebar.number_input("سرمایه اولیه (دلار)", 1000, 1000000, 10000)
put_strike_pct = st.sidebar.slider("درصد قیمت اعمال اختیار فروش (Married Put)", 70, 100, 90)

# --- دریافت داده‌ها ---
@st.cache_data
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna()

data = get_data(tickers, start_date, end_date)
st.subheader("داده‌های تاریخی پرتفو")
st.dataframe(data.tail())

# --- محاسبه بازده و کوواریانس ---
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

# --- نمایش نمودار قیمت ---
st.subheader("نمودار قیمت دارایی‌ها")
fig = go.Figure()
for t in tickers:
    fig.add_trace(go.Scatter(x=data.index, y=data[t], name=t))
st.plotly_chart(fig, use_container_width=True)

# --- شبیه‌سازی مونت‌کارلو ---
def monte_carlo_simulation(mean_returns, cov_matrix, weights, n_sim, n_days, initial_investment):
    sim_results = np.zeros((n_sim, n_days))
    for i in range(n_sim):
        prices = np.ones(n_days) * initial_investment
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
        portfolio_returns = np.dot(daily_returns, weights)
        prices = initial_investment * np.cumprod(1 + portfolio_returns)
        sim_results[i, :] = prices
    return sim_results

sim_results = monte_carlo_simulation(mean_returns, cov_matrix, weights, n_sim, n_days, initial_investment)
sim_df = pd.DataFrame(sim_results.T, index=range(1, n_days+1))

st.subheader("شبیه‌سازی مونت‌کارلو پرتفو")
fig2 = go.Figure()
for i in range(min(100, n_sim)):
    fig2.add_trace(go.Scatter(x=sim_df.index, y=sim_df.iloc[:, i], line=dict(width=1), opacity=0.2, showlegend=False))
fig2.update_layout(title="پرتفو شبیه‌سازی شده", xaxis_title="روز", yaxis_title="ارزش پرتفو")
st.plotly_chart(fig2, use_container_width=True)

# --- محاسبه VaR و CVaR ---
def calculate_var_cvar(sim_results, confidence_level):
    final_values = sim_results[:, -1]
    var = np.percentile(final_values, 100 - confidence_level)
    cvar = final_values[final_values <= var].mean()
    return var, cvar

var, cvar = calculate_var_cvar(sim_results, confidence_level)
st.markdown(f"**Value at Risk (VaR) در سطح {confidence_level}%:** {var:,.0f} دلار")
st.markdown(f"**Conditional Value at Risk (CVaR):** {cvar:,.0f} دلار")

# --- Married Put Strategy ---
st.subheader("مقایسه پرتفو با استراتژی Married Put")
put_strike = (put_strike_pct / 100) * initial_investment
sim_results_put = np.maximum(sim_results, put_strike)
fig3 = go.Figure()
fig3.add_trace(go.Box(y=sim_results[:, -1], name="پرتفو معمولی"))
fig3.add_trace(go.Box(y=sim_results_put[:, -1], name="Married Put"))
fig3.update_layout(title="مقایسه ارزش نهایی پرتفو", yaxis_title="ارزش نهایی")
st.plotly_chart(fig3, use_container_width=True)

st.markdown(f"**حداقل ارزش پرتفو با Married Put:** {put_strike:,.0f} دلار")

# --- دانلود نتایج ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True)
    return output.getvalue()

st.subheader("دانلود نتایج")
excel_data = to_excel(sim_df)
st.download_button("دانلود مسیرهای شبیه‌سازی به اکسل", data=excel_data, file_name="sim_results.xlsx")

# --- پیام‌های راهنما و خطا ---
st.info("برای تحلیل دقیق‌تر می‌توانید تعداد دارایی‌ها، وزن‌ها و پارامترهای شبیه‌سازی را تغییر دهید.")
