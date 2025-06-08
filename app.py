import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import base64

# ... سایر توابع و بخش‌های قبلی فایل بدون تغییر ...

# =========================
# 14. تحلیل پرتفوی، محاسبه، نمودارها و توضیحات هر بخش
# =========================
if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
    prices_df = pd.DataFrame()
    asset_names = []
    for t, df in st.session_state["downloaded_dfs"]:
        name = t
        if "Date" not in df.columns or "Price" not in df.columns:
            continue
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)
    for t, df in st.session_state["uploaded_dfs"]:
        name = t
        if "Date" not in df.columns or "Price" not in df.columns:
            continue
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

    st.subheader("📉 روند قیمت دارایی‌ها")
    st.markdown("""
    <div dir="rtl" style="text-align: right;">
    این نمودار، روند تاریخی قیمت هر دارایی (asset) را در بازه انتخابی نمایش می‌دهد. این به شما کمک می‌کند نوسانات و ریسک دارایی‌ها را بهتر مشاهده کنید.
    اگر بین اعداد انگلیسی و فارسی یا نماد انگلیسی و فارسی فاصله بیفتد، نمایش داده اصلاح می‌شود.
    </div>
    """, unsafe_allow_html=True)
    st.line_chart(prices_df.resample(resample_rule).last().dropna())

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    # --- [تعدیل سری قیمتی دارایی‌ها بر اساس بیمه] ---
    for name in asset_names:
        if name in st.session_state["insured_assets"]:
            info = st.session_state["insured_assets"][name]
            insured = prices_df[name].copy()
            # کف قیمتی بیمه‌شده
            insured[insured < info['strike']] = info['strike']
            # کسر پرمیوم از قیمت ابتدایی (در صورت نیاز)
            insured.iloc[0] -= info['premium']
            prices_df[name] = insured

    try:
        resampled_prices = prices_df.resample(resample_rule).last().dropna()
        returns = resampled_prices.pct_change().dropna()
        mean_returns = returns.mean() * annual_factor
        cov_matrix = returns.cov() * annual_factor
        std_devs = np.sqrt(np.diag(cov_matrix))
        
        # ... بقیه تحلیل و نمودارها عین قبل (بدون تغییر) ...

    except Exception as e:
        st.error(f"خطای تحلیل پرتفو: {e}")

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price یا Close یا Open را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")

# ... ادامه کد ... (بقیه توابع و بخش‌های فایل بدون تغییر)
