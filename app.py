import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import base64

# =========================
# 1. ذخیره داده‌های دانلودی/آپلودی و بیمه در session_state
# =========================
if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "insured_assets" not in st.session_state:
    st.session_state["insured_assets"] = {}

# =========================
# 2. تابع خواندن فایل csv و استخراج price/date
# =========================
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

# =========================
# 3. لینک دانلود دیتافریم به csv
# =========================
def download_link(df, filename):
    csv = df.reset_index(drop=True).to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ دریافت فایل CSV</a>'

# =========================
# 4. تبدیل داده یاهو به فرمت price/date
# =========================
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

# =========================
# 5. محاسبه Max Drawdown پرتفوی
# =========================
def calculate_max_drawdown(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    max_dd = drawdown.min()
    return max_dd

# =========================
# 6. نمایش بازده و ریسک پرتفوی در بازه های مختلف
# =========================
def show_periodic_risk_return(resampled_prices, weights, label):
    pf_prices = (resampled_prices * weights).sum(axis=1)
    pf_returns = pf_prices.pct_change().dropna()
    ann_factor = 12 if resampled_prices.index.freqstr and resampled_prices.index.freqstr.upper().startswith('M') else 52
    mean_ann = pf_returns.mean() * ann_factor
    risk_ann = pf_returns.std() * (ann_factor ** 0.5)
    pf_prices_monthly = pf_prices.resample('M').last().dropna()
    pf_returns_monthly = pf_prices_monthly.pct_change().dropna()
    mean_month = pf_returns_monthly.mean()
    risk_month = pf_returns_monthly.std()
    pf_prices_weekly = pf_prices.resample('W').last().dropna()
    pf_returns_weekly = pf_prices_weekly.pct_change().dropna()
    mean_week = pf_returns_weekly.mean()
    risk_week = pf_returns_weekly.std()
    st.markdown(f"#### 📊 {label}")
    st.markdown(f"""<div dir="rtl" style="text-align:right">
    <b>سالانه:</b> بازده: {mean_ann:.2%} | ریسک: {risk_ann:.2%}<br>
    <b>ماهانه:</b> بازده: {mean_month:.2%} | ریسک: {risk_month:.2%}<br>
    <b>هفتگی:</b> بازده: {mean_week:.2%} | ریسک: {risk_week:.2%}
    </div>
    """, unsafe_allow_html=True)

# =========================
# 7. Efficient Frontier با محدودیت وزن
# =========================
def efficient_frontier(mean_returns, cov_matrix, annual_factor, points=200, min_weights=None, max_weights=None):
    num_assets = len(mean_returns)
    results = np.zeros((3, points))
    weight_record = []
    for i in range(points):
        while True:
            weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
            if min_weights is not None:
                weights = np.maximum(weights, min_weights)
            if max_weights is not None:
                weights = np.minimum(weights, max_weights)
            weights /= np.sum(weights)
            if (min_weights is None or np.all(weights >= min_weights)) and (max_weights is None or np.all(weights <= max_weights)):
                break
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = port_std
        results[1, i] = port_return
        results[2, i] = (port_return) / port_std if port_std > 0 else 0
        weight_record.append(weights)
    return results, np.array(weight_record)

# =========================
# 8. توضیحات ابزار (راهنما)
# =========================
st.markdown("""
<div dir="rtl" style="text-align: right;">
<h3>ابزار تحلیل پرتفو: توضیحات کلی</h3>
این ابزار برای تحلیل و بهینه‌سازی ترکیب دارایی‌های یک پرتفو (Portfolio) با قابلیت‌های حرفه‌ای شبیه‌سازی مونت‌کارلو، بیمه و محدودیت‌های سفارشی طراحی شده است.
</div>
""", unsafe_allow_html=True)

# ========== بارگذاری، تنظیمات، بیمه و محدودیت وزن ==========
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True, key="uploader"
)
if st.session_state["downloaded_dfs"]:
    st.sidebar.markdown("<b>حذف دارایی‌های دانلود شده:</b>", unsafe_allow_html=True)
    for idx, (t, df) in enumerate(st.session_state["downloaded_dfs"]):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.markdown(f"<div dir='rtl' style='text-align: right; font-size: 14px'>{t}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("❌", key=f"delete_dl_{t}_{idx}"):
                st.session_state["downloaded_dfs"].pop(idx)
                st.experimental_rerun()
if st.session_state["uploaded_dfs"]:
    st.sidebar.markdown("<b>حذف دارایی‌های آپلود شده:</b>", unsafe_allow_html=True)
    for idx, (t, df) in enumerate(st.session_state["uploaded_dfs"]):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.markdown(f"<div dir='rtl' style='text-align: right; font-size: 14px'>{t}</div>", unsafe_allow_html=True)
        with col2:
            if st.button("❌", key=f"delete_up_{t}_{idx}"):
                st.session_state["uploaded_dfs"].pop(idx)
                st.experimental_rerun()

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, 0.25, 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)
st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
rf_rate = st.sidebar.number_input("نرخ بازده سالانه دارایی بدون ریسک (سپرده/اوراق، درصد)", 0.0, 100.0, 20.0, 0.1) / 100

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
    except Exception as ex:
        st.error(f"خطا در دریافت داده: {ex}")

if uploaded_files:
    for file in uploaded_files:
        if not hasattr(file, "uploaded_in_session") or not file.uploaded_in_session:
            df = read_csv_file(file)
            if df is not None:
                st.session_state["uploaded_dfs"].append((file.name.split('.')[0], df))
            file.uploaded_in_session = True

all_asset_names = [t for t, _ in st.session_state["downloaded_dfs"]] + [t for t, _ in st.session_state["uploaded_dfs"]]
for name in all_asset_names:
    with st.sidebar.expander(f"⚙️ بیمه برای {name}", expanded=False):
        st.markdown("<div dir='rtl'>بیمه Married Put یعنی اختیار فروش برای دارایی فعال است تا ریزش‌های شدید را پوشش دهد.</div>", unsafe_allow_html=True)
        insured = st.checkbox(f"فعال‌سازی بیمه برای {name}", key=f"insured_{name}")
        if insured:
            loss_percent = st.number_input(f"📉 درصد ضرر معامله پوت برای {name}", 0.0, 100.0, 30.0, step=0.01, key=f"loss_{name}")
            strike = st.number_input(f"🎯 قیمت اعمال پوت برای {name}", 0.0, 1e6, 100.0, step=0.01, key=f"strike_{name}")
            premium = st.number_input(f"💰 قیمت قرارداد پوت برای {name}", 0.0, 1e6, 5.0, step=0.01, key=f"premium_{name}")
            amount = st.number_input(f"📦 مقدار قرارداد برای {name}", 0.0, 1e6, 1.0, step=0.01, key=f"amount_{name}")
            spot_price = st.number_input(f"📌 قیمت فعلی دارایی پایه {name}", 0.0, 1e6, 100.0, step=0.01, key=f"spot_{name}")
            asset_amount = st.number_input(f"📦 مقدار دارایی پایه {name}", 0.0, 1e6, 1.0, step=0.01, key=f"base_{name}")
            st.session_state["insured_assets"][name] = {
                'loss_percent': loss_percent,
                'strike': strike,
                'premium': premium,
                'amount': amount,
                'spot': spot_price,
                'base': asset_amount
            }
        else:
            st.session_state["insured_assets"].pop(name, None)

st.sidebar.markdown("<hr/>", unsafe_allow_html=True)
st.sidebar.markdown("### محدودیت وزن هر دارایی در پرتفو")
min_weights = {}
max_weights = {}
for name in all_asset_names:
    min_weights[name] = st.sidebar.number_input(f"حداقل وزن {name} (%)", 0.0, 100.0, 0.0, 1.0) / 100
    max_weights[name] = st.sidebar.number_input(f"حداکثر وزن {name} (%)", 0.0, 100.0, 100.0, 1.0) / 100

# ========== تحلیل پرتفو ==========
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
    st.line_chart(prices_df.resample(resample_rule).last().dropna())

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    try:
        resampled_prices = prices_df.resample(resample_rule).last().dropna()
        returns = resampled_prices.pct_change().dropna()
        mean_returns = returns.mean() * annual_factor
        cov_matrix = returns.cov() * annual_factor
        std_devs = np.sqrt(np.diag(cov_matrix))

        # ====== افزودن دارایی بدون ریسک ======
        asset_names_rf = asset_names + ["بدون ریسک"]
        mean_returns_rf = np.append(mean_returns.values, rf_rate)
        cov_matrix_rf = np.zeros((len(asset_names_rf), len(asset_names_rf)))
        cov_matrix_rf[:-1, :-1] = cov_matrix.values

        min_w = np.array([min_weights[n] for n in asset_names] + [0.0])
        max_w = np.array([max_weights[n] for n in asset_names] + [1.0])

        n_portfolios = 3000
        n_mc = 1000
        results = np.zeros((5 + len(asset_names_rf), n_portfolios))
        cvar_results = np.zeros((3 + len(asset_names_rf), n_portfolios))
        np.random.seed(42)
        for i in range(n_portfolios):
            while True:
                weights = np.random.random(len(asset_names_rf))
                weights = np.maximum(weights, min_w)
                weights = np.minimum(weights, max_w)
                weights /= np.sum(weights)
                if np.all(weights >= min_w) and np.all(weights <= max_w):
                    break
            sim_returns = np.random.multivariate_normal(mean_returns_rf, cov_matrix_rf, n_mc)
            for j, name in enumerate(asset_names):
                if name in st.session_state["insured_assets"]:
                    info = st.session_state["insured_assets"][name]
                    last_price = resampled_prices[name].iloc[-1]
                    final_prices = last_price * (1 + sim_returns[:, j])
                    put_pnl = np.maximum(info['strike'] - final_prices, 0) * info['amount'] - info['premium'] * info['amount']
                    asset_pnl = (final_prices - last_price) * info['base']
                    total_pnl = asset_pnl + put_pnl
                    sim_returns[:, j] = total_pnl / (last_price * max(info['base'], 1e-8))
            port_mc_returns = np.dot(sim_returns, weights)
            port_return = port_mc_returns.mean()
            port_std = port_mc_returns.std()
            sharpe_ratio = (port_return - rf_rate) / port_std if port_std > 0 else 0
            sortino_ratio = (port_return - rf_rate) / (port_mc_returns[port_mc_returns < 0].std() if np.any(port_mc_returns < 0) else 1)
            VaR = np.percentile(port_mc_returns, (1 - cvar_alpha) * 100)
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

        best_idx = np.argmin(np.abs(results[1] - user_risk))
        best_weights = results[5:, best_idx]
        best_cvar_idx = np.argmin(results[4])
        best_cvar_weights = results[5:, best_cvar_idx]

        st.subheader("📊 داشبورد خلاصه پرتفو")
        show_periodic_risk_return(resampled_prices, best_weights[:-1], "پرتفو بهینه مونت‌کارلو")
        show_periodic_risk_return(resampled_prices, best_cvar_weights[:-1], f"پرتفو بهینه CVaR ({int(cvar_alpha*100)}%)")

        # ====== مرز کارا و MPT ======
        ef_results, ef_weights = efficient_frontier(mean_returns, cov_matrix, annual_factor, points=200,
                                                    min_weights=np.array([min_weights[n] for n in asset_names]),
                                                    max_weights=np.array([max_weights[n] for n in asset_names]))
        max_sharpe_idx_ef = np.argmax(ef_results[2])
        mpt_weights = ef_weights[max_sharpe_idx_ef]

        # نمایش پرتفو MPT
        st.subheader("📊 پرتفو بهینه بر مبنای مرز کارا (MPT)")
        pf_prices_mpt = (resampled_prices * mpt_weights).sum(axis=1)
        pf_returns_mpt = pf_prices_mpt.pct_change().dropna()
        mean_ann_mpt = pf_returns_mpt.mean() * annual_factor
        risk_ann_mpt = pf_returns_mpt.std() * (annual_factor ** 0.5)
        st.markdown(
            f"""<div dir="rtl" style="text-align:right">
            <b>سالانه:</b> بازده: {mean_ann_mpt:.2%} | ریسک: {risk_ann_mpt:.2%}<br>
            <b>وزن پرتفو:</b> {' | '.join([f"{n}: {w:.2%}" for n, w in zip(asset_names, mpt_weights)])}
            </div>
            """, unsafe_allow_html=True)

        # نمودار مرز کارا (ترکیبی)
        fig_all = go.Figure()
        # مرز کارا
        fig_all.add_trace(go.Scatter(
            x=ef_results[0]*100, y=ef_results[1]*100,
            mode='lines+markers', marker=dict(color='gray', size=5), name='مرز کارا (MPT)'
        ))
        # پرتفو بهینه MC
        fig_all.add_trace(go.Scatter(
            x=[results[1, best_idx]*100], y=[results[0, best_idx]*100],
            mode='markers+text', marker=dict(size=14, color='blue', symbol='diamond'),
            text=["MC"], textposition="top right", name='پرتفو بهینه Monte Carlo'
        ))
        # پرتفو بهینه CVaR
        fig_all.add_trace(go.Scatter(
            x=[cvar_results[0, best_cvar_idx]*100], y=[cvar_results[1, best_cvar_idx]*100],
            mode='markers+text', marker=dict(size=14, color='orange', symbol='triangle-up'),
            text=["CVaR"], textposition="top center", name='پرتفو بهینه CVaR'
        ))
        # پرتفو بهینه MPT
        fig_all.add_trace(go.Scatter(
            x=[ef_results[0, max_sharpe_idx_ef]*100], y=[ef_results[1, max_sharpe_idx_ef]*100],
            mode='markers+text', marker=dict(size=16, color='red', symbol='star'),
            text=["MPT"], textposition="bottom right", name='پرتفو بهینه MPT'
        ))
        fig_all.update_layout(
            title="مرز کارا با نمایش پرتفوهای منتخب (MC, CVaR, MPT)",
            xaxis_title="ریسک (%)",
            yaxis_title="بازده (%)"
        )
        st.plotly_chart(fig_all, use_container_width=True)

        # ====== مقایسه هیستوگرام سود/زیان پرتفو با و بدون بیمه ======
        st.subheader("📉 مقایسه توزیع سود/زیان پرتفو قبل و بعد از بیمه (شبیه‌سازی ۱۰٬۰۰۰ مرتبه)")
        n_sim_hist = 10000
        sim_returns_noins = np.random.multivariate_normal(mean_returns_rf, cov_matrix_rf, n_sim_hist)
        port_mc_returns_noins = np.dot(sim_returns_noins, best_weights)
        VaR_noins = np.percentile(port_mc_returns_noins, (1-cvar_alpha)*100)
        cvar_noins = port_mc_returns_noins[port_mc_returns_noins <= VaR_noins].mean()
        loss_prob_noins = np.mean(port_mc_returns_noins < 0)

        sim_returns_ins = sim_returns_noins.copy()
        for j, name in enumerate(asset_names):
            if name in st.session_state["insured_assets"]:
                info = st.session_state["insured_assets"][name]
                last_price = resampled_prices[name].iloc[-1]
                final_prices = last_price * (1 + sim_returns_ins[:, j])
                put_pnl = np.maximum(info['strike'] - final_prices, 0) * info['amount'] - info['premium'] * info['amount']
                asset_pnl = (final_prices - last_price) * info['base']
                total_pnl = asset_pnl + put_pnl
                sim_returns_ins[:, j] = total_pnl / (last_price * max(info['base'], 1e-8))
        port_mc_returns_ins = np.dot(sim_returns_ins, best_weights)
        VaR_ins = np.percentile(port_mc_returns_ins, (1-cvar_alpha)*100)
        cvar_ins = port_mc_returns_ins[port_mc_returns_ins <= VaR_ins].mean()
        loss_prob_ins = np.mean(port_mc_returns_ins < 0)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=port_mc_returns_noins, nbinsx=50, opacity=0.5, name="بدون بیمه", marker_color='red'
        ))
        fig_hist.add_trace(go.Histogram(
            x=port_mc_returns_ins, nbinsx=50, opacity=0.5, name="با بیمه", marker_color='green'
        ))
        fig_hist.add_vline(x=cvar_noins, line_dash="dashdot", line_color="red",
                           annotation_text=f"CVaR بدون بیمه: {cvar_noins:.2%}", annotation_position="bottom left")
        fig_hist.add_vline(x=cvar_ins, line_dash="dashdot", line_color="green",
                           annotation_text=f"CVaR با بیمه: {cvar_ins:.2%}", annotation_position="bottom right")
        fig_hist.add_vline(x=VaR_noins, line_dash="dot", line_color="red", annotation_text="VaR بدون بیمه", annotation_position="top left")
        fig_hist.add_vline(x=VaR_ins, line_dash="dot", line_color="green", annotation_text="VaR با بیمه", annotation_position="top right")
        fig_hist.update_layout(
            barmode='overlay',
            title="مقایسه توزیع سود/زیان پرتفو قبل و بعد از بیمه",
            xaxis_title="بازده شبیه‌سازی‌شده پرتفو",
            yaxis_title="تعداد",
            legend=dict(x=0.7, y=0.95)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown(f"""
        <div dir="rtl" style="text-align:right; font-size:16px">
        <b>CVaR بدون بیمه:</b> <span style="color:red">{cvar_noins:.2%}</span><br>
        <b>CVaR با بیمه:</b> <span style="color:green">{cvar_ins:.2%}</span><br>
        <b>احتمال زیان (بدون بیمه):</b> <span style="color:red">{loss_prob_noins:.2%}</span><br>
        <b>احتمال زیان (با بیمه):</b> <span style="color:green">{loss_prob_ins:.2%}</span>
        </div>
        """, unsafe_allow_html=True)
        st.info("همانطور که در نمودار مشاهده می‌کنید، **با فعال شدن بیمه، انتهای سمت چپ (دم منفی) توزیع سود/زیان کوتاه‌تر شده و احتمال زیان‌های شدید و مقدار CVaR کاهش یافته است. این یعنی بیمه پورتفو ریسک زیان‌های سنگین را کاهش می‌دهد، حتی اگر مقدار انحراف معیار (ریسک کلی) خیلی تغییر نکند.")

    except Exception as e:
        st.error(f"خطای تحلیل پرتفو: {e}")

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price یا Close یا Open را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
