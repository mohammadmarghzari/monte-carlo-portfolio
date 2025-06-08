import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import json

from ui_helpers import format_money, format_percent, format_float, validate_weights, validate_investment_amount, download_link
from data_handlers import read_csv_file, get_price_dataframe_from_yf
from portfolio_analysis import run_portfolio_analysis, run_scenario_analysis, get_portfolio_results_df
from insurance import insurance_input_sidebar, plot_married_put, compute_adjusted_cov
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
if "manual_weights" not in st.session_state:
    st.session_state["manual_weights"] = {}
if "enabled_tickers" not in st.session_state:
    st.session_state["enabled_tickers"] = {}
if "portfolio_config" not in st.session_state:
    st.session_state["portfolio_config"] = {}

st.title("تحلیل و مدیریت پرتفو با امکانات حرفه‌ای")

# ========== Sidebar: File Upload/Delete ==========
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True, key="uploader"
)

if st.session_state["downloaded_dfs"]:
    st.sidebar.markdown("<b>حذف دارایی‌های دانلود شده/مدیریت فعال بودن:</b>", unsafe_allow_html=True)
    for idx, (t, df) in enumerate(st.session_state["downloaded_dfs"]):
        col1, col2, col3 = st.sidebar.columns([4, 1, 1])
        enabled = st.session_state["enabled_tickers"].get(t, True)
        with col1:
            st.markdown(f"<div dir='rtl' style='text-align: right; font-size: 14px'>{t}</div>", unsafe_allow_html=True)
        with col2:
            flag = st.checkbox("فعال", value=enabled, key=f"enable_{t}")
            st.session_state["enabled_tickers"][t] = flag
        with col3:
            if st.button("❌", key=f"delete_dl_{t}_{idx}"):
                st.session_state["insured_assets"].pop(t, None)
                st.session_state["downloaded_dfs"].pop(idx)
                st.session_state["enabled_tickers"].pop(t, None)
                st.experimental_rerun()

if st.session_state["uploaded_dfs"]:
    st.sidebar.markdown("<b>حذف دارایی‌های آپلود شده:</b>", unsafe_allow_html=True)
    for idx, (t, df) in enumerate(st.session_state["uploaded_dfs"]):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.markdown(f"<div dir='rtl' style='text-align: right; font-size: 14px'>{t}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("❌", key=f"delete_up_{t}_{idx}"):
                st.session_state["insured_assets"].pop(t, None)
                st.session_state["uploaded_dfs"].pop(idx)
                st.experimental_rerun()

# ========== Sidebar: Params and Yahoo Download ==========
period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, 0.25, 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

with st.sidebar.expander("📥 دانلود داده آنلاین از Yahoo Finance"):
    st.markdown("""
    <div dir="rtl" style="text-align: right;">
    <b>راهنما:</b>
    <br>نمادها را با کاما و بدون فاصله وارد کنید (مثال: <span style="direction:ltr;display:inline-block">BTC-USD,AAPL,ETH-USD</span>)
    </div>
    """, unsafe_allow_html=True)
    tickers_input = st.text_input("نماد دارایی‌ها (با کاما و بدون فاصله)")
    start = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
    end = st.date_input("تاریخ پایان", value=pd.to_datetime("today"))
    download_btn = st.button("دریافت داده آنلاین")

if download_btn and tickers_input.strip():
    tickers = [t.strip() for t in tickers_input.strip().split(",") if t.strip()]
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
        if data.empty:
            st.error("داده‌ای دریافت نشد!")
        else:
            new_downloaded = []
            for t in tickers:
                df, err = get_price_dataframe_from_yf(data, t)
                if df is not None:
                    df['Date'] = pd.to_datetime(df['Date'])
                    new_downloaded.append((t, df))
                    st.success(f"داده {t} با موفقیت دانلود شد.")
                    st.markdown(download_link(df, f"{t}_historical.csv"), unsafe_allow_html=True)
                else:
                    st.error(f"{err}")
            st.session_state["downloaded_dfs"].extend(new_downloaded)
            for t, _ in new_downloaded:
                st.session_state["enabled_tickers"][t] = True
    except Exception as ex:
        st.error(f"خطا در دریافت داده: {ex}")

if uploaded_files:
    for file in uploaded_files:
        if not hasattr(file, "uploaded_in_session") or not file.uploaded_in_session:
            df = read_csv_file(file)
            if df is not None:
                st.session_state["uploaded_dfs"].append((file.name.split('.')[0], df))
            file.uploaded_in_session = True

# ========== Sidebar: Investment Amount & Insurance ==========
all_asset_names = [t for t, _ in st.session_state["downloaded_dfs"] if st.session_state["enabled_tickers"].get(t, True)] + [t for t, _ in st.session_state["uploaded_dfs"]]

with st.sidebar.expander("💵 مقدار کل سرمایه‌گذاری (معادل دلاری)", expanded=True):
    investment_input = st.text_input("مقدار سرمایه کل (دلار)", value=format_float(st.session_state["investment_amount"]).replace(',', ''), key="inv_amount_inp")
    valid_amount, val = validate_investment_amount(investment_input)
    if not valid_amount:
        st.sidebar.error("مقدار سرمایه معتبر نیست!")
        val = 0
    st.session_state["investment_amount"] = val
    st.markdown(f"مقدار واردشده: <b>{format_money(val)}</b>", unsafe_allow_html=True)

for name in all_asset_names:
    insurance_input_sidebar(name)

# ========== Sidebar: Manual Weights ==========
manual_weights_mode = st.sidebar.checkbox("فعال‌سازی وزن‌دهی دستی", key="manual_weights_mode")
manual_weights = {}
if manual_weights_mode:
    st.sidebar.markdown("وزن هر دارایی را وارد کنید (جمع باید ۱۰۰٪ باشد):")
    total = 0
    for asset_name in all_asset_names:
        manual_weights[asset_name] = st.sidebar.number_input(
            f"وزن {asset_name} (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key=f"w_{asset_name}"
        )
        total += manual_weights[asset_name]
    valid_weights = validate_weights(manual_weights)
    if not valid_weights:
        st.sidebar.error("جمع وزن‌ها باید دقیقاً ۱۰۰٪ باشد و هیچ وزنی منفی نباشد.")
else:
    valid_weights = True

# ========== Sidebar: سناریو و شوک ==========
scenario_mode = st.sidebar.checkbox("فعال‌سازی تحلیل سناریو")
scenario = None
if scenario_mode:
    st.sidebar.markdown("تحلیل شوک بازار:")
    drop_percent = st.sidebar.slider("ریزش کل بازار (%)", 0, 100, 0)
    jump_asset = st.sidebar.selectbox("دارایی با جهش", ["-"] + all_asset_names)
    jump_percent = st.sidebar.slider("درصد جهش این دارایی", 0, 200, 0)
    if drop_percent > 0 or (jump_asset != "-" and jump_percent > 0):
        scenario = {"drop_all": drop_percent, "jump_asset": jump_asset, "jump_percent": jump_percent}

# ========== Sidebar: ذخیره/بازیابی پیکربندی ==========
with st.sidebar.expander("💾 ذخیره/بازیابی پرتفو"):
    if st.button("دانلود تنظیمات پرتفو (JSON)"):
        config = {
            "downloaded_dfs": [t for t, _ in st.session_state["downloaded_dfs"]],
            "enabled_tickers": st.session_state["enabled_tickers"],
            "investment_amount": st.session_state["investment_amount"],
            "insured_assets": st.session_state["insured_assets"],
            "manual_weights": manual_weights if manual_weights_mode else None
        }
        st.download_button("دانلود", save_portfolio_json(config), file_name="portfolio.json")
    uploaded_json = st.file_uploader("آپلود تنظیمات پرتفو", type="json", key="load_json")
    if uploaded_json:
        config = load_portfolio_json(uploaded_json.read().decode())
        st.session_state["enabled_tickers"] = config.get("enabled_tickers", {})
        st.session_state["investment_amount"] = config.get("investment_amount", 1000.0)
        st.session_state["insured_assets"] = config.get("insured_assets", {})
        st.session_state["manual_weights"] = config.get("manual_weights", {})
        st.success("تنظیمات با موفقیت بارگذاری شد! لطفاً صفحه را رفرش کنید.")

# ========== Main Analysis ==========
if all_asset_names and valid_weights:
    # ساخت دیتافریم قیمت‌ها
    prices_df = pd.DataFrame()
    asset_names = []
    for t, df in st.session_state["downloaded_dfs"]:
        if not st.session_state["enabled_tickers"].get(t, True):
            continue
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

    # سناریو
    if scenario:
        prices_df = run_scenario_analysis(prices_df, scenario)

    # تحلیل پرتفو
    analysis = run_portfolio_analysis(
        prices_df,
        asset_names,
        resample_rule,
        annual_factor,
        user_risk,
        cvar_alpha,
        manual_weights if manual_weights_mode and valid_weights else None
    )

    # نمودار دایره‌ای وزن‌دهی
    st.subheader("نمودار دایره‌ای توزیع وزن پرتفو")
    pie_data = analysis["active_weights"]
    st.plotly_chart(go.Figure(data=[
        go.Pie(labels=asset_names, values=pie_data)
    ]), use_container_width=True)

    # خروجی اکسل و PDF
    st.subheader("خروجی اکسل و PDF")
    results_df = get_portfolio_results_df(analysis, asset_names)
    col1, col2 = st.columns(2)
    with col1:
        excel = export_excel(results_df)
        st.download_button("⬇️ دانلود خروجی Excel", excel, file_name="portfolio_analysis.xlsx")
    with col2:
        pdf = export_pdf(results_df)
        st.download_button("⬇️ دانلود خروجی PDF", pdf, file_name="portfolio_analysis.pdf")

    # بیمه
    st.subheader("📉 بیمه دارایی‌ها (Married Put)")
    for name in st.session_state["insured_assets"]:
        if name in asset_names:
            plot_married_put(name)
else:
    st.warning("⚠️ لطفاً فایل‌های داده را بارگذاری و پارامترها را معتبر وارد کنید.")
