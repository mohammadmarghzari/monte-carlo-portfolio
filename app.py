import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
from datetime import datetime

st.set_page_config(page_title="تحلیل پرتفو با بیمه Married Put", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با بیمه Married Put و مرز کارا (MPT)")

# --- توابع کمکی ---
def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.replace('%', '').str.lower()
        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

def calculate_married_put_payoff(spot_price, strike, premium, contract_amount, asset_amount):
    # سود و زیان قرارداد Married Put
    # قرارداد پوت که تا قیمت اعمال (strike) سود داره، و پرمیوم هزینه هست
    # تعداد قرارداد * 100 (اندازه قرارداد) ضرب میشه (استاندارد بازار)
    option_payoff = np.maximum(strike - spot_price, 0) - premium
    total_payoff = option_payoff * contract_amount * 100 + (spot_price - spot_price) * asset_amount
    return total_payoff

def format_float(val):
    # نمایش اعشار تا دو رقم، ولی مقدار اصلی با دقت بیشتر ذخیره شود
    return f"{val:.2f}"

# --- بارگذاری فایل‌ها ---
st.sidebar.header("📂 بارگذاری فایل‌های دارایی (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "هر دارایی یک فایل CSV (Date و Price داشته باشد)", 
    type=["csv"], 
    accept_multiple_files=True
)

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
if period == 'ماهانه':
    resample_rule = 'M'
    annual_factor = 12
elif period == 'سه‌ماهه':
    resample_rule = 'Q'
    annual_factor = 4
else:
    resample_rule = '2Q'
    annual_factor = 2

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []
    asset_data = {}

    for file in uploaded_files:
        df = read_csv_file(file)
        if df is None:
            continue
        name = file.name.split('.')[0]
        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"فایل {name} باید شامل ستون‌های 'Date' و 'Price' باشد.")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price']).set_index('Date')
        df.columns = [name]

        if prices_df.empty:
            prices_df = df
        else:
            prices_df = prices_df.join(df, how='inner')

        asset_names.append(name)
        asset_data[name] = df

    if prices_df.empty:
        st.error("❌ داده معتبری برای تحلیل یافت نشد.")
        st.stop()

    # نمایش خلاصه قیمت‌ها
    st.subheader("📊 پیش‌نمایش قیمت‌های دارایی‌ها")
    st.dataframe(prices_df.tail())

    # بازنمونه‌گیری
    resampled_prices = prices_df.resample(resample_rule).last()
    resampled_prices = resampled_prices.dropna()
    returns = resampled_prices.pct_change().dropna()
    if returns.empty:
        st.error("❌ بازده قابل محاسبه نیست. لطفاً داده‌ها را بررسی کنید.")
        st.stop()

    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    asset_std = np.sqrt(np.diag(cov_matrix))

    # --- بخش بیمه و Married Put برای هر دارایی ---
    st.sidebar.header("🛡️ تنظیمات بیمه Married Put برای هر دارایی")

    insurance_settings = {}
    for name in asset_names:
        st.sidebar.markdown(f"### {name}")
        insure = st.sidebar.checkbox(f"بیمه فعال برای {name}", key=f"insure_{name}")
        if insure:
            premium = st.sidebar.number_input(f"پرمیوم آپشن پوت ({name})", min_value=0.0, step=0.001, format="%.3f", key=f"premium_{name}")
            strike = st.sidebar.number_input(f"قیمت اعمال (Strike) ({name})", min_value=0.0, step=0.01, format="%.2f", key=f"strike_{name}")
            contract_amount = st.sidebar.number_input(f"مقدار قرارداد پوت ({name})", min_value=0.0, step=0.01, format="%.2f", key=f"contract_amount_{name}")
            asset_base_price = st.sidebar.number_input(f"قیمت دارایی پایه ({name})", min_value=0.0, step=0.01, format="%.2f", key=f"asset_base_price_{name}")
            asset_base_amount = st.sidebar.number_input(f"مقدار دارایی پایه ({name})", min_value=0.0, step=0.01, format="%.2f", key=f"asset_base_amount_{name}")
            expiry_date = st.sidebar.date_input(f"تاریخ سررسید آپشن ({name})", key=f"expiry_{name}")

            insurance_settings[name] = {
                'active': True,
                'premium': premium,
                'strike': strike,
                'contract_amount': contract_amount,
                'asset_base_price': asset_base_price,
                'asset_base_amount': asset_base_amount,
                'expiry_date': expiry_date
            }
        else:
            insurance_settings[name] = {'active': False}

    # --- وزن دهی و محاسبه وزن با بیمه ---
    # اول وزن ترجیحی (معکوس ریسک)
    inv_risk = 1 / asset_std
    inv_risk /= inv_risk.sum()

    # وزن اصلاح شده با بیمه (کاهش ریسک نسبت به پرمیوم و مقدار قرارداد)
    weights = np.array(inv_risk)
    adjusted_cov = cov_matrix.copy()
    # کاهش ریسک با توجه به بیمه
    for i, name in enumerate(asset_names):
        setting = insurance_settings[name]
        if setting['active']:
            # فرض کنیم بیمه، ریسک را کاهش می‌دهد متناسب با پرمیوم و مقدار قرارداد
            reduction_factor = 1 - (setting['premium'] * setting['contract_amount'] * 100) / (setting['asset_base_price'] * setting['asset_base_amount'] + 1e-10)
            reduction_factor = max(0.1, reduction_factor)  # حداقل 0.1 برای ریسک
            adjusted_cov.iloc[i, i] *= reduction_factor

    # --- شبیه‌سازی مونت‌کارلو ---
    n_portfolios = 5000
    n_assets = len(asset_names)
    results = np.zeros((3 + n_assets, n_portfolios))

    np.random.seed(42)
    for i in range(n_portfolios):
        random_weights = np.random.random(n_assets)
        random_weights *= weights
        random_weights /= np.sum(random_weights)
        port_return = np.dot(random_weights, mean_returns)
        port_std = np.sqrt(random_weights.T @ adjusted_cov.values @ random_weights)

        # محاسبه سود و زیان Married Put روی هر دارایی فعال
        married_put_profit = 0
        for j, name in enumerate(asset_names):
            setting = insurance_settings[name]
            if setting['active']:
                # برای محاسبه، قیمت‌های احتمالی از توزیع نرمال
                # اینجا فرض ساده‌سازی می‌کنیم، spot_price را برابر قیمت پایه دارایی می‌گیریم
                spot_price = setting['asset_base_price']
                payoff = max(setting['strike'] - spot_price, 0) - setting['premium']
                married_put_profit += payoff * setting['contract_amount'] * 100

        # تاثیر Married Put را به بازده اضافه می‌کنیم (نسبت به اندازه پرتفو)
        port_return += married_put_profit / (np.sum(setting['asset_base_price'] * setting['asset_base_amount']) + 1e-10)

        sharpe_ratio = port_return / port_std if port_std != 0 else 0

        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3:, i] = random_weights

    # انتخاب بهترین پرتفو با نزدیک‌ترین ریسک به 30%
    target_risk = 0.3
    best_idx = np.argmin(np.abs(results[1] - target_risk))
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_weights = results[3:, best_idx]

    st.subheader("📊 نتایج پرتفو پیشنهادی (سالانه)")
    st.markdown(f"""
    - ✅ **بازده مورد انتظار سالانه:** {best_return:.2%}  
    - ⚠️ **ریسک سالانه (انحراف معیار):** {best_risk:.2%}  
    - 🧠 **نسبت شارپ:** {best_sharpe:.2f}
    """)
    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 **وزن {name}:** {best_weights[i]*100:.2f}%")

    # --- رسم نمودار پراکندگی پرتفوها ---
    st.subheader("📈 نمودار پراکندگی پرتفوها (ریسک - بازده)")
    fig = px.scatter(
        x=results[1] * 100,
        y=results[0] * 100,
        color=results[2],
        labels={'x': 'ریسک سالانه (%)', 'y': 'بازده سالانه (%)'},
        color_continuous_scale='Viridis',
        title='شبیه‌سازی پرتفوهای مختلف'
    )
    fig.add_trace(go.Scatter(
        x=[best_risk * 100],
        y=[best_return * 100],
        mode='markers',
        marker=dict(color='red', size=14, symbol='star'),
        name='پرتفوی بهینه'
    ))
    st.plotly_chart(fig, use_container_width=True)

    # --- رسم نمودار Married Put (سود و زیان) برای دارایی‌های بیمه شده ---
    st.subheader("📉 نمودار سود و زیان Married Put (برای دارایی‌های بیمه شده)")

    for name in asset_names:
        setting = insurance_settings[name]
        if setting['active']:
            spot_range = np.linspace(setting['strike'] * 0.5, setting['strike'] * 1.5, 100)
            payoff = np.maximum(setting['strike'] - spot_range, 0) - setting['premium']
            total_payoff = payoff * setting['contract_amount'] * 100 + (spot_range - setting['asset_base_price']) * setting['asset_base_amount']

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=spot_range, y=total_payoff, mode='lines+markers', name=f'Married Put {name}'))
            fig2.update_layout(
                title=f"سود و زیان Married Put برای {name}",
                xaxis_title="قیمت دارایی پایه",
                yaxis_title="سود / زیان",
                template='plotly_white'
            )
            st.plotly_chart(fig2, use_container_width=True)

            # امکان ذخیره عکس نمودار
            if st.button(f"ذخیره نمودار Married Put {name} به عنوان عکس PNG"):
                img_bytes = fig2.to_image(format="png")
                st.download_button(
                    label="دانلود عکس PNG",
                    data=img_bytes,
                    file_name=f"married_put_{name}.png",
                    mime="image/png"
                )

else:
    st.info("لطفاً حداقل یک فایل CSV بارگذاری کنید تا تحلیل آغاز شود.")
