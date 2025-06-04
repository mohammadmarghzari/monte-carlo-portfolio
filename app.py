import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from io import BytesIO

st.set_page_config(page_title="تحلیل پرتفو با مدل‌های پیشرفته", layout="wide")
st.title("📊 ابزار تحلیل پرتفو: مونت‌کارلو، CVaR، Married Put و مدل‌های پیشرفته")

# --- راهنمای استفاده ---
with st.expander("📘 راهنمای دریافت داده آنلاین از یاهو فاینانس"):
    st.markdown("""
    - نماد سهام را به فرمت یاهو فاینانس وارد کنید (مثال: AAPL برای اپل، MSFT برای مایکروسافت).
    - برای بورس تهران نماد را به صورت `نماد.IR` وارد کنید (مثال: فولاد.IR).
    - وزن پرتفو باید بین ۰ و ۱ باشد و مجموع آن‌ها ۱ شود.
    - تعداد شبیه‌سازی و بازه زمانی را بر اساس نیاز تنظیم کنید.
    """)

with st.expander("📗 راهنمای مدل‌های تحلیل پرتفو"):
    st.markdown("""
    **مدل‌های قابل انتخاب:**
    - **مونت‌کارلو:** شبیه‌سازی مسیرهای تصادفی برای پیش‌بینی ارزش آینده پرتفو.
    - **CVaR/VaR:** محاسبه ریسک پرتفو بر اساس زیان‌های محتمل در سطوح اطمینان مختلف.
    - **Married Put:** پوشش ریسک با خرید اختیار فروش برای پرتفو.
    - **Black-Litterman:** ترکیب دیدگاه شخصی با مدل مارکویتز برای تخمین بازده بهینه پرتفو.
    - **Risk Parity:** تخصیص سرمایه بر اساس سهم مساوی هر دارایی در ریسک کل پرتفو.
    - **یادگیری ماشین (Random Forest):** پیش‌بینی بازده پرتفو بر اساس داده‌های تاریخی با الگوریتم جنگل تصادفی.
    - **تحلیل سناریو/استرس تست:** بررسی واکنش پرتفو به شوک‌های بازار و سناریوهای بحرانی.
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

# --- انتخاب مدل ---
model_options = [
    "مونت‌کارلو",
    "CVaR/VaR",
    "Married Put",
    "Black-Litterman",
    "Risk Parity",
    "یادگیری ماشین (Random Forest)",
    "تحلیل سناریو/استرس تست"
]
selected_models = st.sidebar.multiselect("مدل‌های مورد استفاده", model_options, default=model_options[:3])

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

# --- مدل مونت‌کارلو ---
if "مونت‌کارلو" in selected_models:
    def monte_carlo_simulation(mean_returns, cov_matrix, weights, n_sim, n_days, initial_investment):
        sim_results = np.zeros((n_sim, n_days))
        for i in range(n_sim):
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

# --- مدل VaR و CVaR ---
if "CVaR/VaR" in selected_models and "مونت‌کارلو" in selected_models:
    def calculate_var_cvar(sim_results, confidence_level):
        final_values = sim_results[:, -1]
        var = np.percentile(final_values, 100 - confidence_level)
        cvar = final_values[final_values <= var].mean()
        return var, cvar

    var, cvar = calculate_var_cvar(sim_results, confidence_level)
    st.markdown(f"**Value at Risk (VaR) در سطح {confidence_level}%:** {var:,.0f} دلار")
    st.markdown(f"**Conditional Value at Risk (CVaR):** {cvar:,.0f} دلار")

# --- مدل Married Put ---
if "Married Put" in selected_models and "مونت‌کارلو" in selected_models:
    st.subheader("مقایسه پرتفو با استراتژی Married Put")
    put_strike = (put_strike_pct / 100) * initial_investment
    sim_results_put = np.maximum(sim_results, put_strike)
    fig3 = go.Figure()
    fig3.add_trace(go.Box(y=sim_results[:, -1], name="پرتفو معمولی"))
    fig3.add_trace(go.Box(y=sim_results_put[:, -1], name="Married Put"))
    fig3.update_layout(title="مقایسه ارزش نهایی پرتفو", yaxis_title="ارزش نهایی")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown(f"**حداقل ارزش پرتفو با Married Put:** {put_strike:,.0f} دلار")

# --- مدل Black-Litterman ---
if "Black-Litterman" in selected_models:
    st.subheader("مدل Black-Litterman")
    # فرض: بازده بازار برابر با میانگین بازده دارایی‌ها و ریسک بازار برابر با میانگین واریانس
    pi = mean_returns
    tau = 0.05  # پارامتر عدم قطعیت
    P = np.eye(len(tickers))  # فرض دیدگاه‌های مستقل
    Q = mean_returns.values.reshape(-1, 1)  # فرض دیدگاه: بازده هر دارایی برابر با میانگین تاریخی
    omega = np.diag(np.diag(tau * cov_matrix.values))  # عدم قطعیت دیدگاه‌ها

    # فرمول Black-Litterman
    inv = np.linalg.inv(tau * cov_matrix.values)
    middle = np.linalg.inv(P.T @ np.linalg.inv(omega) @ P + inv)
    bl_mean = middle @ (inv @ pi.values.reshape(-1, 1) + P.T @ np.linalg.inv(omega) @ Q)
    bl_weights = bl_mean / np.sum(bl_mean)
    bl_weights = bl_weights.flatten()
    bl_weights_df = pd.DataFrame({"نماد": tickers, "وزن بهینه": bl_weights})
    st.write("وزن‌های بهینه پرتفو بر اساس مدل Black-Litterman:")
    st.dataframe(bl_weights_df)

# --- مدل Risk Parity ---
if "Risk Parity" in selected_models:
    st.subheader("مدل Risk Parity")
    # واریانس هر دارایی
    asset_vols = returns.std()
    inv_vols = 1 / asset_vols
    risk_parity_weights = inv_vols / inv_vols.sum()
    rp_weights_df = pd.DataFrame({"نماد": tickers, "وزن بهینه": risk_parity_weights})
    st.write("وزن‌های بهینه پرتفو بر اساس مدل Risk Parity:")
    st.dataframe(rp_weights_df)

# --- مدل یادگیری ماشین (Random Forest) ---
if "یادگیری ماشین (Random Forest)" in selected_models:
    st.subheader("پیش‌بینی بازده پرتفو با Random Forest")
    # ساخت ویژگی‌ها (lagged returns)
    features = returns.shift(1).dropna()
    target = returns.mean(axis=1).shift(-1).dropna()
    features = features.loc[target.index]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    st.write(f"ضریب تعیین (R2) مدل: {rf.score(X_test, y_test):.2f}")
    pred_df = pd.DataFrame({"بازده واقعی": y_test, "پیش‌بینی مدل": y_pred}, index=y_test.index)
    st.line_chart(pred_df)

# --- تحلیل سناریو و استرس تست ---
if "تحلیل سناریو/استرس تست" in selected_models:
    st.subheader("تحلیل سناریو و استرس تست پرتفو")
    st.markdown("""
    در این بخش می‌توانید تاثیر شوک‌های بازار را بر پرتفو مشاهده کنید.
    """)
    shock = st.slider("درصد شوک منفی به بازده روزانه", -20, 0, -10)
    shocked_returns = returns + (shock / 100)
    shocked_portfolio = (shocked_returns @ weights).cumsum()
    normal_portfolio = (returns @ weights).cumsum()
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=returns.index, y=normal_portfolio, name="پرتفو عادی"))
    fig4.add_trace(go.Scatter(x=returns.index, y=shocked_portfolio, name=f"پرتفو با شوک {shock}%"))
    fig4.update_layout(title="استرس تست پرتفو", yaxis_title="بازده تجمعی")
    st.plotly_chart(fig4, use_container_width=True)

# --- دانلود نتایج مونت‌کارلو ---
if "مونت‌کارلو" in selected_models:
    def to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=True)
        return output.getvalue()

    st.subheader("دانلود نتایج")
    excel_data = to_excel(sim_df)
    st.download_button("دانلود مسیرهای شبیه‌سازی به اکسل", data=excel_data, file_name="sim_results.xlsx")

st.info("برای تحلیل دقیق‌تر می‌توانید مدل‌های مختلف را انتخاب و پارامترها را تغییر دهید.")
