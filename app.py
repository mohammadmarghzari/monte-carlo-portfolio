import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import norm

st.set_page_config(page_title="تحلیل پرتفو با بیمه و MPT", layout="wide")
st.title("📊 تحلیل پرتفو با بیمه مرید پوت و مرز کارا")

# --- توابع کمکی ---

def read_and_clean_csv(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()
        date_col = next((c for c in df.columns if 'date' in c), None)
        price_col = next((c for c in df.columns if 'price' in c), None)
        if not date_col or not price_col:
            st.error(f"فایل {file.name} ستون‌های Date و Price را ندارد.")
            return None
        df = df[[date_col, price_col]]
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Price'] = df['Price'].astype(str).str.replace(',', '').astype(float)
        df = df.dropna(subset=['Price'])
        df = df.set_index('Date').sort_index()
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

def resample_returns(prices, rule):
    prices_r = prices.resample(rule).last().dropna()
    returns = prices_r.pct_change().dropna()
    return returns

def format_float_input(label, key, value=0.0, min_val=0.0, max_val=100.0, step=0.01):
    return st.number_input(label, key=key, min_value=min_val, max_value=max_val, value=value, step=step, format="%.2f")

def format_float_input_small(label, key, value=0.0, min_val=0.0, max_val=100.0, step=0.001):
    # برای مقادیر کوچک تر با دقت سه رقم اعشار
    return st.number_input(label, key=key, min_value=min_val, max_value=max_val, value=value, step=step, format="%.3f")

# --- بارگذاری فایل‌ها ---
st.sidebar.header("📁 آپلود فایل‌های CSV دارایی‌ها (هر فایل یک دارایی)")

uploaded_files = st.sidebar.file_uploader(
    "یک یا چند فایل CSV بارگذاری کنید", 
    type=['csv'], 
    accept_multiple_files=True
)

period = st.sidebar.selectbox("بازه زمانی بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
if period == 'ماهانه':
    rule = 'M'
    annual_factor = 12
elif period == 'سه‌ماهه':
    rule = 'Q'
    annual_factor = 4
else:
    rule = '2Q'
    annual_factor = 2

if not uploaded_files or len(uploaded_files) == 0:
    st.info("لطفا حداقل یک فایل CSV آپلود کنید.")
    st.stop()

# خواندن فایل‌ها و ادغام قیمت‌ها
prices_df = pd.DataFrame()
asset_names = []
for file in uploaded_files:
    df = read_and_clean_csv(file)
    if df is not None:
        name = file.name.split('.')[0]
        df.columns = [name]
        asset_names.append(name)
        if prices_df.empty:
            prices_df = df
        else:
            prices_df = prices_df.join(df, how='inner')

if prices_df.empty:
    st.error("داده‌ی معتبر برای تحلیل یافت نشد.")
    st.stop()

st.subheader("آخرین قیمت‌ها")
st.dataframe(prices_df.tail())

# محاسبه بازده
returns = resample_returns(prices_df, rule)
if returns.empty:
    st.error("بازده قابل محاسبه نیست، داده‌ها کافی نیست.")
    st.stop()

mean_returns = returns.mean() * annual_factor
cov_matrix = returns.cov() * annual_factor

# --- بخش بیمه هر دارایی ---
st.sidebar.header("⚙️ تنظیمات بیمه برای هر دارایی")

insurance_data = {}

for asset in asset_names:
    st.sidebar.markdown(f"### دارایی: {asset}")
    use_insurance = st.sidebar.checkbox(f"فعال‌سازی بیمه برای {asset}", key=f"ins_{asset}")
    if use_insurance:
        loss_percent = st.sidebar.number_input(
            f"درصد ضرر بیمه (مثلاً 30٪) برای {asset}",
            min_value=0.0, max_value=100.0, value=30.0, step=0.01,
            key=f"loss_{asset}", format="%.2f"
        )
        # ورودی‌های استراتژی مرید پوت
        strike_price = st.sidebar.number_input(f"قیمت اعمال (Strike Price) برای {asset}", min_value=0.0, value=1000.0, step=0.01, format="%.2f", key=f"strike_{asset}")
        premium = st.sidebar.number_input(f"قیمت قرارداد (Premium) برای {asset}", min_value=0.0, value=10.0, step=0.01, format="%.2f", key=f"premium_{asset}")
        expiry = st.sidebar.date_input(f"تاریخ سررسید برای {asset}", key=f"expiry_{asset}", value=datetime.today())
        contract_qty = st.sidebar.number_input(f"مقدار قرارداد برای {asset}", min_value=0.0, value=1.0, step=0.01, format="%.2f", key=f"qty_{asset}")
        base_price = st.sidebar.number_input(f"قیمت دارایی پایه برای {asset}", min_value=0.0, value=1000.0, step=0.01, format="%.2f", key=f"baseprice_{asset}")
        base_qty = st.sidebar.number_input(f"مقدار دارایی پایه برای {asset}", min_value=0.0, value=1.0, step=0.01, format="%.2f", key=f"baseqty_{asset}")
        
        insurance_data[asset] = {
            "loss_percent": loss_percent / 100,  # درصد ضرر بیمه به عدد اعشاری
            "strike": strike_price,
            "premium": premium,
            "expiry": expiry,
            "contract_qty": contract_qty,
            "base_price": base_price,
            "base_qty": base_qty,
        }
    else:
        insurance_data[asset] = None

# --- محاسبه وزن بیمه بر اساس ریسک ---
# ریسک هر دارایی (انحراف معیار)
asset_risks = np.sqrt(np.diag(cov_matrix))
risk_sum = sum(asset_risks) if sum(asset_risks) > 0 else 1

weights_insurance = {}
for asset in asset_names:
    if insurance_data[asset]:
        weights_insurance[asset] = asset_risks[asset_names.index(asset)] / risk_sum
    else:
        weights_insurance[asset] = 0.0

# --- شبیه‌سازی پرتفو ---
n_portfolios = 5000
n_assets = len(asset_names)
results = np.zeros((3 + n_assets, n_portfolios))
np.random.seed(42)

for i in range(n_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights)
    
    # Adjust covariance matrix for insurance loss effect (بیمه با کاهش ریسک)
    adj_cov = cov_matrix.copy()
    for idx, asset in enumerate(asset_names):
        ins = insurance_data[asset]
        if ins:
            loss_factor = 1 - ins['loss_percent'] * weights_insurance[asset]
            adj_cov.iloc[idx, idx] *= loss_factor**2
    
    port_return = np.dot(weights, mean_returns)
    port_std = np.sqrt(np.dot(weights.T, np.dot(adj_cov, weights)))
    sharpe = port_return / port_std if port_std != 0 else 0
    
    results[0, i] = port_return
    results[1, i] = port_std
    results[2, i] = sharpe
    results[3:, i] = weights

# --- نمایش بهترین پرتفو بر اساس ریسک هدف ---
target_risk = st.slider("ریسک هدف سالانه (انحراف معیار)", 0.01, 1.0, 0.25, 0.01)
idx_best = np.argmin(np.abs(results[1] - target_risk))

best_return = results[0, idx_best]
best_std = results[1, idx_best]
best_sharpe = results[2, idx_best]
best_weights = results[3:, idx_best]

st.subheader("پرتفو بهینه")
st.write(f"بازده سالانه: {best_return:.2%}")
st.write(f"ریسک سالانه: {best_std:.2%}")
st.write(f"نسبت شارپ: {best_sharpe:.2f}")

for i, asset in enumerate(asset_names):
    st.write(f"{asset}: {best_weights[i]*100:.2f}%")

# --- رسم نمودار پراکندگی پرتفوها ---
fig = px.scatter(
    x=results[1]*100,
    y=results[0]*100,
    color=results[2],
    labels={'x':'ریسک سالانه (%)', 'y':'بازده سالانه (%)'},
    title='شبیه‌سازی پرتفوها',
    color_continuous_scale='Viridis',
    width=900, height=600
)
fig.add_scatter(
    x=[best_std*100], y=[best_return*100],
    mode='markers', marker=dict(color='red', size=15, symbol='star'),
    name='پرتفو بهینه'
)

# --- رسم مرز کارا ---
# مرتب کردن نتایج بر اساس ریسک برای مرز کارا
sorted_idx = np.argsort(results[1])
fig.add_trace(go.Scatter(
    x=results[1, sorted_idx]*100,
    y=results[0, sorted_idx]*100,
    mode='lines',
    name='مرز کارا',
    line=dict(color='orange', width=3)
))
st.plotly_chart(fig)

# --- محاسبه سود و زیان دلاری ---
st.subheader("محاسبه سود و زیان تخمینی (دلار)")
base_amount = st.number_input("مقدار دارایی پایه (واحد)", min_value=0.0, value=1.0)
base_price = st.number_input("قیمت پایه هر واحد (دلار)", min_value=0.0, value=1000.0)

est_profit = base_amount * base_price * best_return
est_loss = base_amount * base_price * best_std

st.write(f"سود تخمینی: {est_profit:,.2f} دلار")
st.write(f"ضرر تخمینی (انحراف معیار): {est_loss:,.2f} دلار")

# --- احتمال بازده در ±۱ انحراف معیار ---
conf_level = 0.68
z = norm.ppf((1 + conf_level) / 2)
lower_bound = best_return - z * best_std
upper_bound = best_return + z * best_std
st.write(f"احتمال بازده در بازه ±۱ انحراف معیار ({int(conf_level*100)}%):")
st.write(f"{lower_bound:.2%} تا {upper_bound:.2%}")

# --- احتمال دراگون (افت شدید) ---
st.subheader("احتمال افت شدید (دراگون)")
threshold = st.number_input("آستانه افت (مثلاً 0 یا -10٪)", value=0.0, step=0.01)
prob_dragon = norm.cdf(threshold, loc=best_return, scale=best_std)
st.write(f"احتمال بازده کمتر از {threshold:.2%}: {prob_dragon*100:.2f}%")

# --- نمودار سود و زیان آپشن برای بیمه‌ها ---
st.subheader("نمودار سود و زیان آپشن‌های بیمه شده")

for asset in asset_names:
    ins = insurance_data[asset]
    if ins:
        st.markdown(f"### دارایی: {asset}")
        # محاسبه سود و زیان آپشن مرید پوت
        S = np.linspace(ins['strike']*0.5, ins['strike']*1.5, 100)
        # سود پوت = max(K - S, 0) - Premium
        put_payoff = np.maximum(ins['strike'] - S, 0) - ins['premium']
        # کل سود زیان = سود پوت * تعداد قرارداد + (S - base_price) * base_qty
        total_payoff = put_payoff * ins['contract_qty'] + (S - ins['base_price']) * ins['base_qty']
        fig_option = go.Figure()
        fig_option.add_trace(go.Scatter(x=S, y=put_payoff*ins['contract_qty'], mode='lines', name='سود زیان آپشن پوت'))
        fig_option.add_trace(go.Scatter(x=S, y=(S - ins['base_price']) * ins['base_qty'], mode='lines', name='سود زیان دارایی پایه'))
        fig_option.add_trace(go.Scatter(x=S, y=total_payoff, mode='lines', name='کل سود زیان'))
        fig_option.update_layout(title=f"نمودار سود و زیان آپشن - {asset}",
                                 xaxis_title="قیمت دارایی پایه",
                                 yaxis_title="سود / زیان (دلار)")
        st.plotly_chart(fig_option, use_container_width=True)
