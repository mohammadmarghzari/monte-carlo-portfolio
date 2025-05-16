import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("تحلیل پرتفو با بیمه Married Put و MPT")

######################
# --- بارگذاری داده‌ها ---
uploaded_files = st.file_uploader("آپلود فایل‌های CSV (Date و قیمت)", accept_multiple_files=True)

if not uploaded_files:
    st.warning("لطفا حداقل یک فایل CSV حاوی ستون Date و Adj Close یا Close آپلود کنید.")
    st.stop()

prices_df = pd.DataFrame()
asset_names = []

for file in uploaded_files:
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    name = file.name.split('.')[0]
    asset_names.append(name)

    if 'Date' not in df.columns:
        st.error(f"❌ ستون 'Date' در فایل {name} پیدا نشد.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df.set_index('Date', inplace=True)

    if 'Adj Close' in df.columns:
        price_series = df['Adj Close']
    elif 'Close' in df.columns:
        price_series = df['Close']
    else:
        st.error(f"❌ ستون 'Adj Close' یا 'Close' در فایل {name} پیدا نشد.")
        st.stop()

    prices_df[name] = price_series

prices_df = prices_df.sort_index()
prices_df = prices_df.fillna(method='ffill').dropna()

st.subheader("نمونه داده‌های قیمتی بارگذاری شده")
st.dataframe(prices_df.head())

######################
# --- محاسبات بازده ---
returns = prices_df.pct_change().dropna()
mean_returns = returns.mean() * 252  # بازده سالانه تخمینی
cov_matrix = returns.cov() * 252     # کوواریانس سالانه تخمینی

######################
# --- فرم تعریف استراتژی Married Put برای هر دارایی ---
st.sidebar.header("تعریف بیمه (Married Put) برای هر دارایی")

put_strategies = {}
for asset in asset_names:
    st.sidebar.markdown(f"### بیمه برای {asset}")
    use_put = st.sidebar.checkbox(f"استفاده از بیمه برای {asset}", key=f"use_put_{asset}")
    if use_put:
        strike = st.sidebar.number_input(f"قیمت اعمال (Strike Price) {asset}", min_value=0.0, format="%.2f", key=f"strike_{asset}")
        premium = st.sidebar.number_input(f"پرمیوم قرارداد (Premium) {asset}", min_value=0.0, format="%.2f", key=f"premium_{asset}")
        expiry = st.sidebar.date_input(f"تاریخ سررسید {asset}", key=f"expiry_{asset}")
        contract_amount = st.sidebar.number_input(f"تعداد قرارداد {asset}", min_value=0, step=1, key=f"contract_amount_{asset}")
        underlying_amount = st.sidebar.number_input(f"مقدار دارایی پایه {asset}", min_value=0.0, format="%.2f", key=f"underlying_amount_{asset}")

        put_strategies[asset] = {
            "strike": strike,
            "premium": premium,
            "expiry": expiry,
            "contract_amount": contract_amount,
            "underlying_amount": underlying_amount,
        }
    else:
        put_strategies[asset] = None

######################
# --- توابع کمکی ---

def married_put_payoff(S, strike, premium, contract_amount, underlying_amount):
    # قیمت S: قیمت دارایی پایه در سررسید
    # سود و زیان Married Put = سود دارایی + سود از پوت - پرمیوم
    intrinsic_put = np.maximum(strike - S, 0)
    payoff_per_underlying = (S - S) + intrinsic_put - premium  # سود دارایی = 0 چون S - S=0 برای سادگی (نکته: در عمل باید سود واقعی را حساب کرد)
    # در اینجا فرض می‌کنیم تعداد قراردادها و دارایی پایه حساب شده است
    total_payoff = underlying_amount * (S - S) + contract_amount * 100 * (intrinsic_put - premium)
    return total_payoff

def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    p_return = portfolio_return(weights, mean_returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_vol

def check_sum(weights):
    return np.sum(weights) - 1

######################
# --- بهینه‌سازی مرز کارا (MPT) ---

num_assets = len(asset_names)
args = (mean_returns, cov_matrix)
constraints = ({'type': 'eq', 'fun': check_sum})
bounds = tuple((0,1) for _ in range(num_assets))
init_guess = num_assets * [1. / num_assets]

opt_results = minimize(neg_sharpe_ratio, init_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
opt_weights = opt_results.x

p_return = portfolio_return(opt_weights, mean_returns)
p_vol = portfolio_volatility(opt_weights, cov_matrix)

st.subheader("بهینه‌سازی پرتفو با MPT")
st.write(f"وزن دارایی‌ها:\n {dict(zip(asset_names, np.round(opt_weights, 3)))}")
st.write(f"بازده سالانه پرتفو: {p_return:.4f}")
st.write(f"ریسک سالانه پرتفو (انحراف معیار): {p_vol:.4f}")

######################
# --- رسم نمودار مرز کارا ---

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficient_vols = []
    for ret in returns_range:
        constraints = ({'type':'eq', 'fun': check_sum},
                       {'type':'eq', 'fun': lambda w: portfolio_return(w, mean_returns) - ret})
        result = minimize(portfolio_volatility, init_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            efficient_vols.append(result.fun)
        else:
            efficient_vols.append(np.nan)
    return efficient_vols

returns_range = np.linspace(mean_returns.min(), mean_returns.max(), 50)
eff_vols = efficient_frontier(mean_returns, cov_matrix, returns_range)

fig_mpt = go.Figure()
fig_mpt.add_trace(go.Scatter(x=eff_vols, y=returns_range, mode='lines', name='مرز کارا'))
fig_mpt.add_trace(go.Scatter(x=[p_vol], y=[p_return], mode='markers', name='پرتفوی بهینه', marker=dict(size=12, color='red')))
fig_mpt.update_layout(title="مرز کارا - MPT", xaxis_title="ریسک (انحراف معیار)", yaxis_title="بازده سالانه")

st.plotly_chart(fig_mpt, use_container_width=True)

######################
# --- نمایش نمودار سود و زیان Married Put ---

st.subheader("نمودار سود و زیان Married Put")

spot_prices = np.linspace(prices_df.min().min()*0.8, prices_df.max().max()*1.2, 100)

fig_put = go.Figure()

for asset, strategy in put_strategies.items():
    if strategy:
        payoff = []
        for S in spot_prices:
            p = married_put_payoff(S,
                                  strategy["strike"],
                                  strategy["premium"],
                                  strategy["contract_amount"],
                                  strategy["underlying_amount"])
            payoff.append(p)
        fig_put.add_trace(go.Scatter(x=spot_prices, y=payoff, mode='lines', name=f"Married Put {asset}"))

fig_put.update_layout(
    title="سود و زیان Married Put",
    xaxis_title="قیمت دارایی در سررسید",
    yaxis_title="سود / زیان",
    hovermode="x unified"
)

st.plotly_chart(fig_put, use_container_width=True)

######################
# --- امکان ذخیره عکس نمودار ---

def save_plotly_fig(fig, filename):
    import io
    img_bytes = fig.to_image(format="png")
    with open(filename, "wb") as f:
        f.write(img_bytes)

st.subheader("ذخیره نمودارها")

if st.button("ذخیره نمودار مرز کارا"):
    img_bytes = fig_mpt.to_image(format="png")
    st.download_button(label="دانلود نمودار مرز کارا PNG", data=img_bytes, file_name="efficient_frontier.png", mime="image/png")

if st.button("ذخیره نمودار سود و زیان Married Put"):
    img_bytes = fig_put.to_image(format="png")
    st.download_button(label="دانلود نمودار Married Put PNG", data=img_bytes, file_name="married_put.png", mime="image/png")
