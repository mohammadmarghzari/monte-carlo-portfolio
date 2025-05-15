import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="تحلیل پرتفو با آپشن و مرز کارا", layout="wide")
st.title("📈 تحلیل پرتفو با بیمه آپشن، مونت‌کارلو و مرز کارا")

st.sidebar.header("📂 فایل‌های CSV قیمت")
uploaded_files = st.sidebar.file_uploader("هر دارایی یک فایل CSV با ستون‌های Date و Price", type=['csv'], accept_multiple_files=True)

analysis_mode = st.sidebar.radio("مدل تحلیل پرتفو:", ["مونت‌کارلو (MC)", "مرز کارا (MPT)"])
period = st.sidebar.selectbox("بازه تحلیل بازده:", ['روزانه', 'ماهانه', 'سه‌ماهه'])
if period == 'روزانه':
    resample_rule, annual_factor = 'D', 252
elif period == 'ماهانه':
    resample_rule, annual_factor = 'M', 12
else:
    resample_rule, annual_factor = 'Q', 4

target_risk_slider = st.sidebar.slider("🎯 ریسک هدف برای مرز کارا (%)", 1.0, 50.0, 25.0, step=0.1) / 100
use_put_option = st.sidebar.checkbox("📉 فعال‌سازی بیمه با آپشن پوت")

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []

    for file in uploaded_files:
        name = file.name.split('.')[0]
        
        try:
            # خواندن فایل با در نظر گرفتن encoding مناسب و حذف BOM
            df = pd.read_csv(file, encoding='utf-8-sig')
            
            # پاکسازی نام ستون‌ها
            df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", '').str.strip()
            
            # بررسی وجود ستون‌های مورد نیاز
            if 'Date' not in df.columns or 'Price' not in df.columns:
                st.error(f"فایل {name} باید شامل ستون‌های 'Date' و 'Price' باشد. ستون‌های یافت‌شده: {df.columns.tolist()}")
                continue
                
            # انتخاب و پاکسازی داده‌ها
            df = df[['Date', 'Price']].copy()
            df.dropna(subset=['Date', 'Price'], inplace=True)
            
            # تبدیل تاریخ به فرمت datetime
            try:
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  # برای فرمت روز/ماه/سال
            except:
                df['Date'] = pd.to_datetime(df['Date'])  # تلاش با فرمت پیش‌فرض
            
            df = df.dropna(subset=['Date'])
            df.set_index('Date', inplace=True)
            
            # تغییر نام ستون قیمت به نام دارایی
            df = df[['Price']].rename(columns={'Price': name})
            
            # ادغام با dataframe اصلی
            if prices_df.empty:
                prices_df = df
            else:
                prices_df = prices_df.join(df, how='inner')
                
            asset_names.append(name)
            
        except Exception as e:
            st.error(f"خطا در پردازش فایل {name}: {str(e)}")
            continue

    if not prices_df.empty:
        # محاسبه بازده‌ها
        returns = prices_df.resample(resample_rule).last().pct_change().dropna()
        mean_returns = returns.mean() * annual_factor
        cov_matrix = returns.cov() * annual_factor
        asset_std = np.sqrt(np.diag(cov_matrix))

        base_amounts, base_prices, option_contracts, option_strikes, option_premiums = {}, {}, {}, {}, {}
        coverage = {}

        if use_put_option:
            st.header("🛡 تنظیمات بیمه آپشن برای هر دارایی")
            for asset in asset_names:
                st.markdown(f"#### 📌 {asset}")
                base_amounts[asset] = st.number_input(f"مقدار دارایی پایه - {asset}", 0.0, 1e8, 1.0, step=0.01, format="%.6f", key=f"amount_{asset}")
                base_prices[asset] = st.number_input(f"قیمت خرید دارایی - {asset}", 0.0, 1e6, float(prices_df[asset].iloc[-1]), step=0.01, format="%.6f", key=f"price_{asset}")
                option_contracts[asset] = st.number_input(f"تعداد قرارداد آپشن - {asset}", 0.0, 1e6, 0.0, step=0.0001, format="%.6f", key=f"contracts_{asset}")
                option_strikes[asset] = st.number_input(f"قیمت اعمال - {asset}", 0.0, 1e6, float(prices_df[asset].iloc[-1] * 0.9), step=0.01, format="%.6f", key=f"strike_{asset}")
                option_premiums[asset] = st.number_input(f"قیمت آپشن - {asset}", 0.0, 1e6, float(prices_df[asset].iloc[-1] * 0.05), step=0.01, format="%.6f", key=f"premium_{asset}")

                insured_value = option_contracts[asset] * option_strikes[asset]
                base_value = base_amounts[asset] * base_prices[asset] + 1e-10
                coverage[asset] = min(insured_value / base_value, 1.0)

            adj_returns = mean_returns.copy()
            for asset in asset_names:
                pnl = max(0, option_strikes[asset] - base_prices[asset]) * option_contracts[asset] - option_premiums[asset] * option_contracts[asset]
                pnl_percent = pnl / (base_amounts[asset] * base_prices[asset] + 1e-10)
                adj_returns[asset] += pnl_percent

            adj_cov = cov_matrix.copy()
            for i, a1 in enumerate(asset_names):
                for j, a2 in enumerate(asset_names):
                    c1 = coverage.get(a1, 0)
                    c2 = coverage.get(a2, 0)
                    adj_cov.iloc[i, j] *= (1 - (c1 + c2) / 2) ** 1.5
        else:
            adj_returns = mean_returns
            adj_cov = cov_matrix

        st.header("📊 نتایج پرتفو")
        n = 10000 if analysis_mode == "مونت‌کارلو (MC)" else 500
        results = np.zeros((3 + len(asset_names), n)

        for i in range(n):
            weights = np.random.random(len(asset_names))
            weights /= np.sum(weights)
            port_return = np.dot(weights, adj_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(adj_cov, weights)))
            sharpe = port_return / port_std if port_std > 0 else 0
            results[0, i] = port_return
            results[1, i] = port_std
            results[2, i] = sharpe
            results[3:, i] = weights

        idx_best = np.argmin(np.abs(results[1] - target_risk_slider))
        ret, risk, sharpe = results[0, idx_best], results[1, idx_best], results[2, idx_best]
        weights = results[3:, idx_best]

        st.markdown(f"""
        - ✅ بازده مورد انتظار: {ret:.2%}
        - ⚠️ ریسک سالانه: {risk:.2%}
        - 🧠 نسبت شارپ: {sharpe:.2f}
        """)
        for i, name in enumerate(asset_names):
            st.markdown(f"🔸 وزن {name}: {weights[i]*100:.2f}%")

        fig = px.scatter(x=results[1]*100, y=results[0]*100, color=results[2],
            labels={'x': 'Risk (%)', 'y': 'Expected Return (%)'},
            title=f"Efficient Frontier - {analysis_mode}", color_continuous_scale='Viridis')
        fig.add_trace(go.Scatter(x=[risk*100], y=[ret*100], mode='markers', marker=dict(color='red', size=12, symbol='star'), name='Target'))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("هیچ داده معتبری پس از پردازش فایل‌ها یافت نشد.")
else:
    st.warning("لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید.")
