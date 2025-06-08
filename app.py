import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

# --- 1. Session State ---
if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "insured_assets" not in st.session_state:
    st.session_state["insured_assets"] = {}

# --- 2. Read CSV ---
def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        date_col = [c for c in df.columns if str(c).strip().lower() == 'date'][0]
        price_candidates = [c for c in df.columns if str(c).strip().lower() in ['price', 'close', 'adj close', 'open']]
        price_col = price_candidates[0]
        df = df[[date_col, price_col]].dropna()
        df = df.rename(columns={date_col: "Date", price_col: "Price"})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل: {e}")
        return None

# --- 3. Yahoo Finance Extract ---
def get_price_dataframe_from_yf(data, t):
    if isinstance(data.columns, pd.MultiIndex):
        if t in data.columns.levels[0]:
            df_t = data[t].reset_index()
            price_col = None
            for col in ['Close', 'Adj Close', 'Open']:
                if col in df_t.columns:
                    price_col = col
                    break
            df = df_t[['Date', price_col]].rename(columns={price_col: 'Price'})
            return df
    else:
        if 'Date' not in data.columns:
            data = data.reset_index()
        price_col = None
        for col in ['Close', 'Adj Close', 'Open']:
            if col in data.columns:
                price_col = col
                break
        df = data[['Date', price_col]].rename(columns={price_col: 'Price'})
        return df

# --- 4. Manage Delete Asset ---
def delete_asset(asset_key, source):
    if source == "downloaded":
        st.session_state["downloaded_dfs"] = [item for item in st.session_state["downloaded_dfs"] if f"downloaded_{item[0]}" != asset_key]
    else:
        st.session_state["uploaded_dfs"] = [item for item in st.session_state["uploaded_dfs"] if f"uploaded_{item[0]}" != asset_key]
    st.session_state["insured_assets"].pop(asset_key, None)

# --- 5. Sidebar Upload/Download/Remove ---
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True
)
if uploaded_files:
    for file in uploaded_files:
        df = read_csv_file(file)
        if df is not None and f"uploaded_{file.name.split('.')[0]}" not in [k for k in st.session_state["insured_assets"]]:
            st.session_state["uploaded_dfs"].append((file.name.split('.')[0], df))

with st.sidebar.expander("📥 دانلود داده آنلاین از Yahoo Finance"):
    tickers_input = st.text_input("نماد دارایی‌ها (مثال: BTC-USD,AAPL)")
    start = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
    end = st.date_input("تاریخ پایان", value=pd.to_datetime("today"))
    download_btn = st.button("دریافت داده آنلاین")
if download_btn and tickers_input.strip():
    tickers = [t.strip() for t in tickers_input.strip().split(",") if t.strip()]
    data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
    for t in tickers:
        df = get_price_dataframe_from_yf(data, t)
        if df is not None and f"downloaded_{t}" not in [k for k in st.session_state["insured_assets"]]:
            df['Date'] = pd.to_datetime(df['Date'])
            st.session_state["downloaded_dfs"].append((t, df))

# --- 6. Delete Buttons ---
st.sidebar.markdown("### حذف دارایی")
for t, df in st.session_state["downloaded_dfs"]:
    asset_key = f"downloaded_{t}"
    if st.sidebar.button(f"❌ حذف {t} (دانلود)", key=f"del_downloaded_{t}"):
        delete_asset(asset_key, "downloaded")
        st.experimental_rerun()
for t, df in st.session_state["uploaded_dfs"]:
    asset_key = f"uploaded_{t}"
    if st.sidebar.button(f"❌ حذف {t} (آپلود)", key=f"del_uploaded_{t}"):
        delete_asset(asset_key, "uploaded")
        st.experimental_rerun()

# --- 7. Sidebar Insurance Settings ---
all_asset_names = []
asset_sources = []
for t, df in st.session_state["downloaded_dfs"]:
    all_asset_names.append(t)
    asset_sources.append("downloaded")
for t, df in st.session_state["uploaded_dfs"]:
    all_asset_names.append(t)
    asset_sources.append("uploaded")

for name, source in zip(all_asset_names, asset_sources):
    asset_key = f"{source}_{name}"
    with st.sidebar.expander(f"⚙️ بیمه برای {name}", expanded=False):
        insured = st.checkbox(f"فعال‌سازی بیمه برای {name}", key=f"insured_{asset_key}")
        if insured:
            strike = st.number_input(f"🎯 قیمت اعمال پوت برای {name}", 0.0, 1e8, 100.0, step=0.01, key=f"strike_{asset_key}")
            premium = st.number_input(f"💰 قیمت قرارداد پوت برای {name}", 0.0, 1e8, 5.0, step=0.01, key=f"premium_{asset_key}")
            amount = st.number_input(f"📦 مقدار قرارداد برای {name}", 0.0, 1e6, 1.0, step=0.01, key=f"amount_{asset_key}")
            spot_price = st.number_input(f"📌 قیمت فعلی دارایی پایه {name}", 0.0, 1e8, 100.0, step=0.01, key=f"spot_{asset_key}")
            asset_amount = st.number_input(f"📦 مقدار دارایی پایه {name}", 0.0, 1e6, 1.0, step=0.01, key=f"base_{asset_key}")
            st.session_state["insured_assets"][asset_key] = {
                'strike': strike,
                'premium': premium,
                'amount': amount,
                'spot': spot_price,
                'base': asset_amount,
                'name': name
            }
        else:
            st.session_state["insured_assets"].pop(asset_key, None)

# --- 8. Portfolio Construction ---
if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
    prices_df = pd.DataFrame()
    for (t, df), source in zip(st.session_state["downloaded_dfs"] + st.session_state["uploaded_dfs"], ['downloaded']*len(st.session_state["downloaded_dfs"]) + ['uploaded']*len(st.session_state["uploaded_dfs"])):
        asset_key = f"{source}_{t}"
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [asset_key]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')

    st.subheader("📉 روند قیمت دارایی‌ها")
    st.line_chart(prices_df)

    # مسیر بیمه‌شده
    adjusted_prices_df = prices_df.copy()
    for asset_key, info in st.session_state["insured_assets"].items():
        S = prices_df[asset_key]
        strike = info['strike']
        premium = info['premium']
        amount = info['amount']
        put_profit = np.maximum(strike - S, 0) * amount - premium * amount
        insured_price = S + put_profit
        adjusted_prices_df[asset_key] = insured_price

    # ---------- Portfolio Analytics (Monte Carlo & CVaR) ----------
    resample_rule = 'M'
    annual_factor = 12
    resampled_prices = adjusted_prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))

    n_portfolios = 2500
    n_mc = 500
    results = np.zeros((5 + len(adjusted_prices_df.columns), n_portfolios))
    cvar_results = np.zeros((3 + len(adjusted_prices_df.columns), n_portfolios))
    np.random.seed(42)
    rf = 0
    downside = returns.copy()
    downside[downside > 0] = 0
    for i in range(n_portfolios):
        weights = np.random.random(len(adjusted_prices_df.columns))
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        downside_risk = np.sqrt(np.dot(weights.T, np.dot(downside.cov() * annual_factor, weights)))
        sharpe_ratio = (port_return - rf) / port_std
        sortino_ratio = (port_return - rf) / downside_risk if downside_risk > 0 else np.nan
        mc_sims = np.random.multivariate_normal(mean_returns/annual_factor, cov_matrix/annual_factor, n_mc)
        port_mc_returns = np.dot(mc_sims, weights)
        VaR = np.percentile(port_mc_returns, 5)
        CVaR = port_mc_returns[port_mc_returns <= VaR].mean() if np.any(port_mc_returns <= VaR) else VaR
        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3, i] = sortino_ratio
        results[4, i] = -CVaR
        results[5:, i] = weights
        cvar_results[0, i] = port_std
        cvar_results[1, i] = port_return
        cvar_results[2, i] = -CVaR
        cvar_results[3:, i] = weights

    best_idx = np.argmin(np.abs(results[1] - 0.25))  # ریسک هدف
    best_weights = results[5:, best_idx]
    best_cvar_idx = np.argmin(results[4])
    best_cvar_weights = results[5:, best_cvar_idx]

    def show_periodic_risk_return(resampled_prices, weights, label):
        pf_prices = (resampled_prices * weights).sum(axis=1)
        pf_returns = pf_prices.pct_change().dropna()
        ann_factor = 12
        mean_ann = pf_returns.mean() * ann_factor
        risk_ann = pf_returns.std() * (ann_factor ** 0.5)
        pf_prices_monthly = pf_prices.resample('M').last().dropna()
        pf_returns_monthly = pf_prices_monthly.pct_change().dropna()
        mean_month = pf_returns_monthly.mean()
        risk_month = pf_returns_monthly.std()
        st.markdown(f"#### {label}")
        st.markdown(f"""<div dir="rtl" style="text-align:right">
        <b>سالانه:</b> بازده: {mean_ann:.2%} | ریسک: {risk_ann:.2%}<br>
        <b>ماهانه:</b> بازده: {mean_month:.2%} | ریسک: {risk_month:.2%}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 داشبورد خلاصه پرتفو")
    show_periodic_risk_return(resampled_prices, best_weights, "پرتفو بهینه مونت‌کارلو")
    show_periodic_risk_return(resampled_prices, best_cvar_weights, f"پرتفو بهینه CVaR (5%)")

    # ---------- Married Put Chart ----------
    st.subheader("📉 بیمه دارایی‌ها (Married Put) با نقطه سر به سر")
    for asset_key, info in st.session_state["insured_assets"].items():
        spot = info['spot']
        strike = info['strike']
        premium = info['premium']
        amount = info['amount']
        base = info['base']
        name = info['name']
        x = np.linspace(spot * 0.5, spot * 1.5, 201)
        asset_pnl = (x - spot) * base
        put_pnl = np.where(x < strike, (strike - x) * amount, 0) - premium * amount
        total_pnl = asset_pnl + put_pnl
        percent_pnl = total_pnl / (spot * base) * 100
        cross_idx = np.where(np.diff(np.sign(total_pnl)) != 0)[0]
        break_even_idx = cross_idx[0] + 1 if len(cross_idx) > 0 else np.abs(total_pnl).argmin()
        break_even_price = x[break_even_idx]
        colors = np.where(total_pnl >= 0, 'green', 'red')
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=x,
            y=total_pnl,
            marker_color=colors,
            hovertemplate=
                'قیمت دارایی: %{x:,.0f}<br>' +
                'سود/زیان کل: %{y:,.0f}<br>' +
                'درصد سود/زیان: %{customdata:.2f}%',
            customdata=percent_pnl,
            name='سود/زیان کل'
        ))
        fig2.add_vline(
            x=break_even_price,
            line_dash="dot",
            line_color="blue",
            annotation_text=f"Break-even: {break_even_price:,.0f}",
            annotation_position="top left"
        )
        fig2.add_trace(go.Scatter(
            x=x, y=asset_pnl,
            mode='lines', name='دارایی پایه', line=dict(dash='dot', color='gray')
        ))
        fig2.add_trace(go.Scatter(
            x=x, y=put_pnl,
            mode='lines', name='پوت', line=dict(dash='dot', color='blue')
        ))
        fig2.update_layout(
            title=f"Married Put {name} - نقطه سر به سر و درصد سود/زیان",
            xaxis_title="قیمت دارایی",
            yaxis_title="سود/زیان",
            hovermode="x unified"
        )
        st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price یا Close یا Open را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
