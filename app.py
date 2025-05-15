import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import csv

st.set_page_config(page_title="تحلیل پرتفو کامل", layout="wide")
st.title("📈 تحلیل پرتفو با بیمه آپشن، مونت‌کارلو، سود و زیان تخمینی")

st.sidebar.header("📂 بارگذاری فایل‌های CSV")
uploaded_files = st.sidebar.file_uploader(
    "آپلود فایل‌های CSV (شامل ستون‌های تاریخ و قیمت)", 
    type=["csv"], 
    accept_multiple_files=True
)

period = st.sidebar.selectbox("بازه تحلیل:", ['روزانه', 'ماهانه', 'سه‌ماهه'])
resample_rule, annual_factor = {'روزانه': ('D', 252), 'ماهانه': ('M', 12), 'سه‌ماهه': ('Q', 4)}[period]

use_option = st.sidebar.checkbox("📉 استفاده از بیمه با آپشن پوت")

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []

    for file in uploaded_files:
        name = file.name.split('.')[0]
        try:
            # خواندن فایل CSV با مدیریت نقل‌قول‌ها
            df = pd.read_csv(
                file,
                thousands=',',
                quoting=csv.QUOTE_ALL,  # مدیریت نقل‌قول‌ها
                quotechar='"',          # کاراکتر نقل‌قول
                skipinitialspace=True   # حذف فاصله‌های اضافی
            )

            # پاکسازی نام ستون‌ها
           df.columns = df.columns.str.strip().str.lower().str.replace('"', '', regex=False)
            # نمایش ستون‌های خام برای دیباگ
            st.write(f"ستون‌های فایل {name}: {list(df.columns)}")

            # بررسی وجود ستون‌های تاریخ و قیمت
            date_col = None
            price_col = None
            if 'date' in df.columns and 'price' in df.columns:
                date_col, price_col = 'date', 'price'
            elif 'timeopen' in df.columns and 'close' in df.columns:
                date_col, price_col = 'timeopen', 'close'
            else:
                st.error(f"فایل {name} باید شامل ستون‌های 'date' و 'price' یا 'timeopen' و 'close' باشه. ستون‌های موجود: {list(df.columns)}")
                continue

            # انتخاب ستون‌های مورد نیاز
            df = df[[date_col, price_col]].copy()

            # تبدیل ستون تاریخ به datetime با فرمت‌های مختلف
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
            if df[date_col].isna().all():
                # امتحان فرمت‌های دیگر
                for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d']:
                    df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors='coerce', utc=True)
                    if not df[date_col].isna().all():
                        break
                if df[date_col].isna().all():
                    st.error(f"ستون تاریخ ({date_col}) در فایل {name} به درستی به datetime تبدیل نشد. لطفاً فرمت تاریخ رو چک کن (مثال: '2024-05-15' یا '15/05/2024').")
                    continue

            # تبدیل ستون قیمت به عددی
            df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

            # حذف ردیف‌های نامعتبر
            df.dropna(subset=[date_col, price_col], inplace=True)

            if df.empty:
                st.error(f"فایل {name} پس از پردازش هیچ داده معتبری نداره.")
                continue

            # تنظیم ایندکس
            df.set_index(date_col, inplace=True)
            df.rename(columns={price_col: name}, inplace=True)

            if prices_df.empty:
                prices_df = df
            else:
                prices_df = prices_df.join(df, how='inner')

            asset_names.append(name)
        except Exception as e:
            st.error(f"خطا در پردازش فایل {name}: {e}")
            continue

    if prices_df.empty:
        st.error("❌ هیچ داده معتبری برای تحلیل وجود نداره.")
        st.stop()

    # بررسی نوع ایندکس قبل از resample
    if not pd.api.types.is_datetime64_any_dtype(prices_df.index):
        st.error("⛔ ایندکس دیتافریم از نوع datetime نیست. لطفاً فرمت ستون‌های تاریخ رو چک کن.")
        st.stop()

    st.subheader("🧾 پیش‌نمایش داده‌های قیمت")
    st.dataframe(prices_df.tail())

    try:
        prices_resampled = prices_df.resample(resample_rule).last().dropna()
        returns = prices_resampled.pct_change().dropna()
        if returns.empty:
            st.error("❌ داده‌های کافی برای محاسبه بازده وجود نداره.")
            st.stop()
        mean_returns = returns.mean() * annual_factor
        cov_matrix = returns.cov() * annual_factor
    except Exception as e:
        st.error(f"خطا در بازنمونه‌برداری یا محاسبه بازده: {e}")
        st.stop()

    option_data = {}
    if use_option:
        st.subheader("🛡 تنظیمات بیمه با آپشن پوت")
        for asset in asset_names:
            st.markdown(f"#### {asset}")
            amount = st.number_input(f"🔹 مقدار دارایی پایه - {asset}", 0.0, 1e6, 1.0, 0.01, key=f"amt_{asset}")
            buy_price = st.number_input(f"💵 قیمت خرید دارایی - {asset}", 0.0, 1e6, 1000.0, 0.01, key=f"bp_{asset}")
            contracts = st.number_input(f"📄 تعداد قرارداد آپشن - {asset}", 0.0, 1e6, 0.0, 0.0001, key=f"opt_{asset}")
            strike = st.number_input(f"🎯 قیمت اعمال - {asset}", 0.0, 1e6, 1000.0, 0.01, key=f"strike_{asset}")
            premium = st.number_input(f"💰 قیمت هر آپشن - {asset}", 0.0, 1e6, 50.0, 0.01, key=f"premium_{asset}")
            base_val = amount * buy_price
            insured_val = contracts * strike
            coverage = min(insured_val / base_val, 1.0) if base_val > 0 else 0
            pnl = max(0, strike - buy_price) * contracts - contracts * premium
            option_data[asset] = {
                "coverage": coverage,
                "pnl_ratio": pnl / base_val if base_val > 0 else 0
            }
        adj_returns = mean_returns + np.array([option_data[a]['pnl_ratio'] for a in asset_names])
        avg_coverage = np.mean([option_data[a]['coverage'] for a in asset_names])
        adj_cov = cov_matrix * (1 - avg_coverage) ** 2
    else:
        adj_returns = mean_returns
        adj_cov = cov_matrix

    n_assets = len(asset_names)
    n_portfolios = 5000
    results = np.zeros((3 + n_assets, n_portfolios))
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        ret = np.dot(weights, adj_returns)
        risk = np.sqrt(np.dot(weights.T, np.dot(adj_cov, weights)))
        sharpe = ret / risk if risk != 0 else 0
        results[0, i] = ret
        results[1, i] = risk
        results[2, i] = sharpe
        results[3:, i] = weights

    # انتخاب پرتفو با حداکثر نسبت شارپ
    idx = np.argmax(results[2])
    best_ret, best_risk, best_sharpe = results[0, idx], results[1, idx], results[2, idx]
    best_weights = results[3:, idx]

    st.subheader("📊 نتایج پرتفو پیشنهادی")
    st.markdown(f"✅ بازده سالانه: {best_ret:.2%}")
    st.markdown(f"⚠️ ریسک سالانه: {best_risk:.2%}")
    st.markdown(f"🧠 نسبت شارپ: {best_sharpe:.2f}")
    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 وزن {name}: {best_weights[i]*100:.2f}%")

    fig = px.scatter(
        x=results[1] * 100,
        y=results[0] * 100,
        color=results[2],
        labels={"x": "ریسک (%)", "y": "بازده مورد انتظار (%)"},
        title="مرز کارا",
        color_continuous_scale="Viridis"
    )
    fig.add_trace(go.Scatter(
        x=[best_risk * 100],
        y=[best_ret * 100],
        mode="markers",
        marker=dict(color="red", size=12, symbol="star"),
        name="پرتفو بهینه"
    ))
    st.plotly_chart(fig)

    st.subheader("💵 سود و زیان دلاری تخمینی")
    capital = st.number_input("💰 مقدار سرمایه‌گذاری (دلار)", 0.0, 1e9, 10000.0, 100.0)
    st.success(f"📈 سود تخمینی: {best_ret * capital:,.2f} دلار")
    st.error(f"📉 زیان احتمالی (±1σ): {-best_risk * capital:,.2f} دلار")

    st.subheader("🎯 بازه بازده با احتمال 68%")
    low = best_ret - best_risk
    high = best_ret + best_risk
    st.info(f"درصدی: از {low:.2%} تا {high:.2%}")
    st.info(f"دلاری: از {capital * low:,.2f} تا {capital * high:,.2f}")
else:
    st.info("لطفاً فایل‌های CSV با ستون‌های 'date' و 'price' یا 'timeopen' و 'close' آپلود کنید.")
