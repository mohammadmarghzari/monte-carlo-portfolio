import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

from portfolio_analysis import (
    run_portfolio_analysis,
    plot_pie_charts,
    plot_efficient_frontiers,
    plot_mpt_efficient_frontier,
    price_forecast_section
)

# ===== Helper Functions =====
def format_money(val):
    if val == 0:
        return "۰ دلار"
    elif val >= 1:
        return "{:,.0f} دلار".format(val)
    else:
        return "{:.3f} دلار".format(val).replace('.', '٫')

def format_percent(val):
    return "{:.3f}%".format(val*100).replace('.', '٫')

def format_float(val):
    if abs(val) >= 1:
        return "{:,.3f}".format(val).rstrip('0').rstrip('.')
    else:
        return "{:.6f}".format(val).rstrip('0').rstrip('.')

def read_csv_file(file):
    try:
        file.seek(0)
        df_try = pd.read_csv(file)
        cols_lower = [str(c).strip().lower() for c in df_try.columns]
        if any(x in cols_lower for x in ['date']):
            df = df_try.copy()
        else:
            file.seek(0)
            df = pd.read_csv(file, header=None)
            header_idx = None
            for i in range(min(5, len(df))):
                row = [str(x).strip().lower() for x in df.iloc[i].tolist()]
                if any('date' == x for x in row):
                    header_idx = i
                    break
            if header_idx is None:
                raise Exception("سطر عنوان مناسب (شامل date) یافت نشد.")
            header_row = df.iloc[header_idx].tolist()
            df = df.iloc[header_idx+1:].reset_index(drop=True)
            df.columns = header_row

        date_col = [c for c in df.columns if str(c).strip().lower() == 'date']
        if not date_col:
            raise Exception("ستون تاریخ با نام 'Date' یا مشابه آن یافت نشد.")
        date_col = date_col[0]
        price_candidates = [c for c in df.columns if str(c).strip().lower() in ['price', 'close', 'adj close', 'open']]
        if not price_candidates:
            price_candidates = [c for c in df.columns if c != date_col]
        if not price_candidates:
            raise Exception("ستون قیمت مناسب یافت نشد.")
        price_col = price_candidates[0]
        df = df[[date_col, price_col]].dropna()
        if df.empty:
            raise Exception("پس از حذف داده‌های خالی، داده‌ای باقی نماند.")

        df = df.rename(columns={date_col: "Date", price_col: "Price"})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        if df.empty:
            raise Exception("پس از تبدیل نوع داده، داده معتبری باقی نماند.")
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

def get_price_dataframe_from_yf(data, t):
    if isinstance(data.columns, pd.MultiIndex):
        if t in data.columns.levels[0]:
            df_t = data[t].reset_index()
            price_col = None
            for col in ['Close', 'Adj Close', 'Open']:
                if col in df_t.columns:
                    price_col = col
                    break
            if price_col is None:
                return None, f"هیچ یک از ستون‌های قیمت (Close, Adj Close, Open) برای {t} پیدا نشد."
            df = df_t[['Date', price_col]].rename(columns={price_col: 'Price'})
            return df, None
        else:
            return None, f"نماد {t} در داده‌های دریافتی وجود ندارد."
    else:
        if 'Date' not in data.columns:
            data = data.reset_index()
        price_col = None
        for col in ['Close', 'Adj Close', 'Open']:
            if col in data.columns:
                price_col = col
                break
        if price_col is None:
            return None, f"هیچ یک از ستون‌های قیمت (Close, Adj Close, Open) برای {t} پیدا نشد."
        df = data[['Date', price_col]].rename(columns={price_col: 'Price'})
        return df, None

# ========== Session State ==========
if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "investment_amount" not in st.session_state:
    st.session_state["investment_amount"] = 1000.0

st.title("تحلیل پیشرفته پرتفو و پیش‌بینی قیمت")

# ========== Sidebar ==========
st.sidebar.header("بارگذاری داده")
uploaded_files = st.sidebar.file_uploader(
    "فایل CSV دارایی‌ها را بارگذاری کنید", type=['csv'], accept_multiple_files=True, key="uploader"
)

with st.sidebar.expander("دریافت داده آنلاین از Yahoo Finance"):
    tickers_input = st.text_input("نماد دارایی‌ها (با کاما)")
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
                else:
                    st.error(f"{err}")
            st.session_state["downloaded_dfs"].extend(new_downloaded)
    except Exception as ex:
        st.error(f"خطا در دریافت داده: {ex}")

if uploaded_files:
    for file in uploaded_files:
        if not hasattr(file, "uploaded_in_session") or not file.uploaded_in_session:
            df = read_csv_file(file)
            if df is not None:
                st.session_state["uploaded_dfs"].append((file.name.split('.')[0], df))
            file.uploaded_in_session = True

investment_amount = st.sidebar.number_input(
    "مقدار کل سرمایه (دلار)", min_value=0.0, value=st.session_state["investment_amount"], step=100.0
)
st.session_state["investment_amount"] = investment_amount

# ========== Main Logic ==========
all_asset_names = [t for t, _ in st.session_state["downloaded_dfs"]] + [t for t, _ in st.session_state["uploaded_dfs"]]

if all_asset_names:
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

    analysis = run_portfolio_analysis(prices_df, asset_names, investment_amount)

    # نمودار دایره‌ای برای همه سبک‌ها
    st.subheader("📊 نمودار دایره‌ای توزیع وزنی پرتفو برای همه سبک‌ها")
    plot_pie_charts(analysis, asset_names, investment_amount)

    # نمودار مرز کارا همه سبک‌ها
    st.subheader("📈 نمودار مرز کارا (Efficient Frontier) همه سبک‌ها")
    plot_efficient_frontiers(analysis, asset_names)

    # نمودار مرز کارا ویژه MPT
    st.subheader("📈 نمودار مرز کارا (MPT) با نمایش پرتفوهای منتخب")
    plot_mpt_efficient_frontier(analysis, asset_names)

    # پیش‌بینی قیمت آینده هر دارایی
    st.subheader("🔮 پیش‌بینی پیشرفته قیمت آینده دارایی‌ها")
    price_forecast_section(prices_df, asset_names)

else:
    st.warning("⚠️ لطفاً داده بارگذاری کنید.")
