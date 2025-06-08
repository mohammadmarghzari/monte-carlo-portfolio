import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

from ui_helpers import format_money, format_percent, format_float, download_link
from data_handlers import read_csv_file, get_price_dataframe_from_yf
from portfolio_analysis import (
    run_portfolio_analysis,
    plot_pie_charts,
    plot_efficient_frontiers,
    plot_mpt_efficient_frontier,
    price_forecast_section,
)
from insurance import insurance_input_sidebar, plot_married_put
from persistence import save_portfolio_json, load_portfolio_json
from export import export_excel, export_pdf

# ========== Session State ==========
if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "insured_assets" not in st.session_state:
    st.session_state["insured_assets"] = {}
if "investment_amount" not in st.session_state:
    st.session_state["investment_amount"] = 1000.0

st.title("تحلیل و مدیریت پرتفو با امکانات حرفه‌ای")

# ... [سایر بخش‌های sidebar مثل قبل] ...

# ========== Main Analysis ==========
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

    # تحلیل پرتفو و خروجی وزن‌ها و مرز کارا
    analysis = run_portfolio_analysis(prices_df, asset_names, ...)

    # ۱. نمودار دایره‌ای (Pie) برای هر سبک
    st.subheader("📊 نمودار دایره‌ای (Pie) توزیع وزنی پرتفو برای همه سبک‌ها")
    plot_pie_charts(analysis, asset_names, st.session_state["investment_amount"])

    # ۲. نمودار مرز کارا برای همه سبک‌ها
    st.subheader("📈 نمودار مرز کارا (Efficient Frontier) همه سبک‌ها")
    plot_efficient_frontiers(analysis, asset_names)

    # ۳. نمودار مرز کارا ویژه سبک MPT (با مارکر پرتفوهای خاص)
    st.subheader("📈 نمودار مرز کارا (MPT) با نمایش پرتفوهای منتخب")
    plot_mpt_efficient_frontier(analysis, asset_names)

    # ۴. پیش‌بینی قیمت آینده هر دارایی
    st.subheader("🔮 پیش‌بینی پیشرفته قیمت آینده دارایی‌ها")
    price_forecast_section(prices_df, asset_names)

    # سایر خروجی‌ها و بیمه و...
    # ...
else:
    st.warning("⚠️ لطفاً فایل‌های داده را بارگذاری و پارامترها را معتبر وارد کنید.")