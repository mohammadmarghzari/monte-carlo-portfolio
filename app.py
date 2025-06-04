import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from io import BytesIO
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

st.set_page_config(page_title="تحلیل پرتفو (بارگذاری چند فایل داده)", layout="wide")
st.title("📊 ابزار تحلیل پرتفو (بارگذاری چند فایل داده تاریخی)")

with st.expander("📘 راهنمای بارگذاری داده"):
    st.markdown("""
    - می‌توانید چندین فایل داده (CSV یا TXT) برای نمادهای مختلف بارگذاری کنید.
    - هر فایل باید شامل تاریخ و قیمت دارایی باشد (ستون‌هایی مثل date, price, close, adj close و ...).
    - ابزار به طور خودکار نام ستون‌ها را تمیز و داده‌های ناقص را حذف می‌کند و همه داده‌ها را در یک جدول ترکیب می‌کند.
    - نام هر نماد از نام فایل گرفته می‌شود (مثلاً AAPL.csv).
    """)

uploaded_files = st.file_uploader("فایل‌های داده پرتفو را بارگذاری کنید (CSV یا TXT)", type=["csv", "txt"], accept_multiple_files=True)
if not uploaded_files:
    st.warning("لطفاً حداقل یک فایل داده بارگذاری کنید.")
    st.stop()

def clean_columns(columns):
    return [re.sub(r'[^a-zA-Z0-9]', '', str(col)).lower() for col in columns]

def auto_detect_price_column(cols):
    price_keywords = ['price', 'close', 'adjclose', 'adj_close', 'last']
    for col in cols:
        for key in price_keywords:
            if key in col:
                return col
    return None

all_data = []

for file in uploaded_files:
    # خواندن فایل
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_csv(file, delimiter="\t")
    df.columns = clean_columns(df.columns)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    # پیدا کردن ستون تاریخ و قیمت
    date_col = None
    for col in df.columns:
        if 'date' in col or 'data' in col or 'time' in col:
            date_col = col
            break
    price_col = auto_detect_price_column(df.columns)
    if date_col is None or price_col is None:
        st.error(f"ستون تاریخ یا قیمت در فایل {file.name} پیدا نشد.")
        st.stop()
    df = df[[date_col, price_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    # نام نماد را از نام فایل استخراج کن
    symbol = file.name.split('.')[0]
    df = df.rename(columns={price_col: symbol, date_col: 'date'})
    all_data.append(df.set_index('date'))

# ادغام همه داده‌ها بر اساس تاریخ
final_df = pd.concat(all_data, axis=1)
final_df = final_df.dropna(how='all')  # حذف ردیف‌هایی که همه ستون‌ها خالی‌اند
final_df = final_df.sort_index()

st.subheader("داده‌های ترکیب‌شده و تمیزشده:")
st.dataframe(final_df)

tickers = list(final_df.columns)
st.sidebar.header("تنظیمات پرتفو")
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

n_sim = st.sidebar.number_input("تعداد شبیه‌سازی مونت‌کارلو", 100, 5000, 1000)
n_days = st.sidebar.number_input("تعداد روزهای پیش‌بینی", 30, 365, 180)
confidence_level = st.sidebar.slider("سطح اطمینان VaR/CVaR", 90, 99, 95)
initial_investment = st.sidebar.number_input("سرمایه اولیه (دلار)", 1000, 1000000, 10000)
put_strike_pct = st.sidebar.slider("درصد قیمت اعمال اختیار فروش (Married Put)", 70, 100, 90)

model_options = [
    "مونت‌کارلو",
    "CVaR/VaR",
    "Married Put",
    "Black-Litterman",
    "Risk Parity",
    "یادگیری ماشین (Random Forest)",
    "یادگیری ماشین (LightGBM)",
    "یادگیری ماشین (XGBoost)",
    "تحلیل سناریو/استرس تست"
]
selected_models = st.sidebar.multiselect("مدل‌های مورد استفاده", model_options, default=model_options[:3])

returns = final_df.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

st.subheader("نمودار قیمت دارایی‌ها")
fig = go.Figure()
for t in final_df.columns:
    fig.add_trace(go.Scatter(x=final_df.index, y=final_df[t], name=t))
st.plotly_chart(fig, use_container_width=True)

# مونت‌کارلو
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

if "CVaR/VaR" in selected_models and "مونت‌کارلو" in selected_models:
    def calculate_var_cvar(sim_results, confidence_level):
        final_values = sim_results[:, -1]
        var = np.percentile(final_values, 100 - confidence_level)
        cvar = final_values[final_values <= var].mean()
        return var, cvar

    var, cvar = calculate_var_cvar(sim_results, confidence_level)
    st.markdown(f"**Value at Risk (VaR) در سطح {confidence_level}%:** {var:,.0f} دلار")
    st.markdown(f"**Conditional Value at Risk (CVaR):** {cvar:,.0f} دلار")

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

if "Black-Litterman" in selected_models:
    st.subheader("مدل Black-Litterman")
    pi = mean_returns
    tau = 0.05
    P = np.eye(len(final_df.columns))
    Q = mean_returns.values.reshape(-1, 1)
    omega = np.diag(np.diag(tau * cov_matrix.values))
    inv = np.linalg.inv(tau * cov_matrix.values)
    middle = np.linalg.inv(P.T @ np.linalg.inv(omega) @ P + inv)
    bl_mean = middle @ (inv @ pi.values.reshape(-1, 1) + P.T @ np.linalg.inv(omega) @ Q)
    bl_weights = bl_mean / np.sum(bl_mean)
    bl_weights = bl_weights.flatten()
    bl_weights_df = pd.DataFrame({"نماد": final_df.columns, "وزن بهینه": bl_weights})
    st.write("وزن‌های بهینه پرتفو بر اساس مدل Black-Litterman:")
    st.dataframe(bl_weights_df)

if "Risk Parity" in selected_models:
    st.subheader("مدل Risk Parity")
    asset_vols = returns.std()
    inv_vols = 1 / asset_vols
    risk_parity_weights = inv_vols / inv_vols.sum()
    rp_weights_df = pd.DataFrame({"نماد": final_df.columns, "وزن بهینه": risk_parity_weights})
    st.write("وزن‌های بهینه پرتفو بر اساس مدل Risk Parity:")
    st.dataframe(rp_weights_df)

def ml_feature_engineering(returns):
    features = returns.shift(1).dropna()
    target = returns.mean(axis=1).shift(-1).dropna()
    features = features.loc[target.index]
    return features, target

if "یادگیری ماشین (Random Forest)" in selected_models:
    st.subheader("پیش‌بینی بازده پرتفو با Random Forest")
    features, target = ml_feature_engineering(returns)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    st.write(f"ضریب تعیین (R2) مدل: {rf.score(X_test, y_test):.2f}")
    pred_df = pd.DataFrame({"بازده واقعی": y_test, "پیش‌بینی مدل": y_pred}, index=y_test.index)
    st.line_chart(pred_df)

if "یادگیری ماشین (LightGBM)" in selected_models:
    st.subheader("پیش‌بینی بازده پرتفو با LightGBM")
    features, target = ml_feature_engineering(returns)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    lgbm = LGBMRegressor(n_estimators=100, random_state=42)
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    st.write(f"ضریب تعیین (R2) مدل: {lgbm.score(X_test, y_test):.2f}")
    pred_df = pd.DataFrame({"بازده واقعی": y_test, "پیش‌بینی مدل": y_pred}, index=y_test.index)
    st.line_chart(pred_df)

if "یادگیری ماشین (XGBoost)" in selected_models:
    st.subheader("پیش‌بینی بازده پرتفو با XGBoost")
    features, target = ml_feature_engineering(returns)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    st.write(f"ضریب تعیین (R2) مدل: {xgb.score(X_test, y_test):.2f}")
    pred_df = pd.DataFrame({"بازده واقعی": y_test, "پیش‌بینی مدل": y_pred}, index=y_test.index)
    st.line_chart(pred_df)

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
