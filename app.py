import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# پیکربندی صفحه
st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو و استراتژی Married Put")

# ---------- SIDEBAR با فرم ----------
with st.sidebar.form("settings_form"):
    st.header("📂 بارگذاری و تنظیمات")
    uploaded_files = st.file_uploader(
        "چند فایل CSV آپلود کنید (هر دارایی یک فایل)",
        type=['csv'], accept_multiple_files=True)
    period = st.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
    
    # --- جدید: ریسک هدف پرتفوی ---
    target_risk_input = st.number_input(
        "⚖️ ریسک هدف پرتفوی سالیانه (%)",
        min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    
    submitted = st.form_submit_button("اعمال تغییرات")

if not submitted:
    st.info("⚠️ لطفاً فایل‌ها و بازه تحلیل را در سایدبار انتخاب و اعمال کنید.")
    st.stop()

resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]

# ---------- تابع خواندن فایل ----------
def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

if not uploaded_files:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید.")
    st.stop()

prices_df = pd.DataFrame()
asset_names = []
insured_assets = {}

for file in uploaded_files:
    df = read_csv_file(file)
    if df is None:
        continue

    name = file.name.split('.')[0]

    if 'Date' not in df.columns or 'Price' not in df.columns:
        st.warning(f"فایل {name} باید دارای ستون‌های 'Date' و 'Price' باشد.")
        continue

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Price'] = df['Price'].astype(str).str.replace(',', '')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Date', 'Price'])
    df = df[['Date', 'Price']].set_index('Date')
    df.columns = [name]

    prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
    asset_names.append(name)

# --- جدید: دریافت ریسک ثابت هر دارایی (با مقدار پیش‌فرض 20 درصد) ---
st.sidebar.header("⚙️ تنظیمات ریسک دارایی‌ها (ثابت سالیانه)")
fixed_risks = {}
for name in asset_names:
    fixed_risks[name] = st.sidebar.number_input(
        f"ریسک سالیانه ثابت برای {name} (%)", 
        min_value=0.0, max_value=100.0, value=20.0, step=0.1, key=f"risk_{name}"
    )

# ---------- تنظیمات بیمه به صورت تب جداگانه در سایدبار ----------
with st.sidebar.expander("⚙️ تنظیمات بیمه (Married Put)"):
    for name in asset_names:
        insured = st.checkbox(f"📌 فعال‌سازی بیمه برای {name}", key=f"insured_{name}")
        if insured:
            loss_percent = st.number_input(f"📉 درصد ضرر معامله پوت برای {name}", 0.0, 100.0, 30.0, step=0.01, key=f"loss_{name}")
            strike = st.number_input(f"🎯 قیمت اعمال پوت برای {name}", 0.0, 1e6, 100.0, step=0.01, key=f"strike_{name}")
            premium = st.number_input(f"💰 قیمت قرارداد پوت برای {name}", 0.0, 1e6, 5.0, step=0.01, key=f"premium_{name}")
            amount = st.number_input(f"📦 مقدار قرارداد برای {name}", 0.0, 1e6, 1.0, step=0.01, key=f"amount_{name}")
            spot_price = st.number_input(f"📌 قیمت فعلی دارایی پایه برای {name}", 0.0, 1e6, 100.0, step=0.01, key=f"spot_{name}")
            asset_amount = st.number_input(f"📦 مقدار دارایی پایه برای {name}", 0.0, 1e6, 1.0, step=0.01, key=f"base_{name}")
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

# ---------- تحلیل بازده ----------
resampled_prices = prices_df.resample(resample_rule).last().dropna()
returns = resampled_prices.pct_change().dropna()
mean_returns = returns.mean() * annual_factor
cov_matrix = returns.cov() * annual_factor

# --- جدید: بجای محاسبه ریسک واقعی، ریسک هر دارایی را ثابت می‌گیریم ---
# برای این کار، ماتریس کوواریانس رو میسازیم با فرض استقلال دارایی‌ها
fixed_std_devs = np.array([fixed_risks[name] / 100 for name in asset_names])
adjusted_cov = np.diag(fixed_std_devs**2)

# اگر بخواهی می‌تونی کوواریانس واقعی رو در ادامه هم دخیل کنی (مثلاً میانگین بگیری)
# اما در اینجا فقط ریسک ثابت را مدنظر گرفتیم

preference_weights = []

for i, name in enumerate(asset_names):
    if name in insured_assets:
        risk_scale = 1 - insured_assets[name]['loss_percent'] / 100
        adjusted_cov[i, i] *= risk_scale**2  # ریسک متناسب با بیمه کمتر می‌شود (توان ۲ برای واریانس)
        preference_weights.append(1 / (fixed_std_devs[i] * risk_scale))
    else:
        preference_weights.append(1 / fixed_std_devs[i])

preference_weights = np.array(preference_weights)
preference_weights /= np.sum(preference_weights)

# ---------- شبیه‌سازی مونت‌کارلو ----------
n_portfolios = 10000
results = np.zeros((3 + len(asset_names), n_portfolios))
np.random.seed(42)

target_risk = target_risk_input / 100  # تبدیل به عدد بین 0 و 1

for i in range(n_portfolios):
    weights = np.random.random(len(asset_names)) * preference_weights
    weights /= np.sum(weights)
    port_return = np.dot(weights, mean_returns)
    port_std = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))
    sharpe_ratio = port_return / port_std if port_std != 0 else 0
    results[0, i] = port_return
    results[1, i] = port_std
    results[2, i] = sharpe_ratio
    results[3:, i] = weights

# پیدا کردن پرتفوی با ریسک نزدیک به ریسک هدف
best_idx = np.argmin(np.abs(results[1] - target_risk))
best_return = results[0, best_idx]
best_risk = results[1, best_idx]
best_sharpe = results[2, best_idx]
best_weights = results[3:, best_idx]

# ---------- نمایش داشبورد کارت ----------
st.subheader("📈 نتایج پرتفو بهینه")
col1, col2, col3 = st.columns(3)
col1.metric("📈 بازده سالانه", f"{best_return:.2%}")
col2.metric("⚠️ ریسک سالانه", f"{best_risk:.2%}")
col3.metric("🧠 نسبت شارپ", f"{best_sharpe:.2f}")

for i, name in enumerate(asset_names):
    st.markdown(f"🔹 وزن {name}: **{best_weights[i]*100:.2f}%**")

# ---------- نمودار پرتفو ----------
fig = px.scatter(x=results[1]*100, y=results[0]*100, color=results[2],
                 labels={'x': 'ریسک (%)', 'y': 'بازده (%)'},
                 title='پرتفوهای شبیه‌سازی‌شده', color_continuous_scale='Viridis')
fig.add_trace(go.Scatter(x=[best_risk*100], y=[best_return*100],
                         mode='markers', marker=dict(size=12, color='red', symbol='star'),
                         name='پرتفوی بهینه'))
st.plotly_chart(fig)

# ---------- تب‌بندی Married Put برای بیمه‌ها ----------
if insured_assets:
    st.subheader("📉 نمودار سود و زیان Married Put")
    tabs = st.tabs([f"{name}" for name in insured_assets])
    for i, (name, info) in enumerate(insured_assets.items()):
        with tabs[i]:
            x = np.linspace(info['spot'] * 0.5, info['spot'] * 1.5, 200)
            asset_pnl = (x - info['spot']) * info['base']
            put_pnl = np.where(x < info['strike'], (info['strike'] - x) * info['amount'], 0) - info['premium'] * info['amount']
            total_pnl = asset_pnl + put_pnl

            fig2 = make_subplots(rows=1, cols=3, subplot_titles=("Married Put", "دارایی پایه", "آپشن پوت"))
            fig2.add_trace(go.Scatter(x=x, y=total_pnl, name='سود و زیان Married Put'), row=1, col=1)
            fig2.add_trace(go.Scatter(x=x, y=asset_pnl, name='دارایی پایه'), row=1, col=2)
            fig2.add_trace(go.Scatter(x=x, y=put_pnl, name='آپشن پوت'), row=1, col=3)

            fig2.update_layout(height=400, width=900, showlegend=True)
            st.plotly_chart(fig2)
else:
    st.info("⚠️ هیچ بیمه‌ای برای دارایی‌ها فعال نشده است.")

# ---------- پیام وضعیت پایین صفحه ----------
st.markdown("---")
st.info("✅ تحلیل کامل انجام شد. برای تحلیل بهتر، داده‌ها را به‌روز نگه دارید.")
