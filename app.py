import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import base64

# --- Session state ---
if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "insured_assets" not in st.session_state:
    st.session_state["insured_assets"] = {}

# --- CSV Reader ---
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

# --- Download Link ---
def download_link(df, filename):
    csv = df.reset_index(drop=True).to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ دریافت فایل CSV</a>'

# --- Yahoo Finance DataFrame Extractor ---
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

# --- Max Drawdown ---
def calculate_max_drawdown(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    return drawdown.min()

# --- Efficient Frontier ---
def efficient_frontier(mean_returns, cov_matrix, annual_factor, points=200):
    num_assets = len(mean_returns)
    results = np.zeros((3, points))
    weight_record = []
    for i in range(points):
        weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = port_std
        results[1,i] = port_return
        results[2,i] = (port_return) / port_std if port_std > 0 else 0
        weight_record.append(weights)
    return results, np.array(weight_record)

# --- Periodic Risk/Return ---
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

# --- توضیحات ---
st.markdown("""
<div dir="rtl" style="text-align: right;">
<h3>ابزار تحلیل پرتفو: توضیحات کلی</h3>
ابزاری برای تحلیل و بهینه‌سازی پرتفو به روش‌های مدرن و کلاسیک!
</div>
""", unsafe_allow_html=True)

# --- بارگذاری/حذف دارایی‌ها ---
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

# --- پارامترها ---
period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, 0.25, 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

# --- دانلود داده یاهو ---
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

# --- آپلود فایل csv و افزودن به لیست دارایی‌ها ---
if uploaded_files:
    for file in uploaded_files:
        if not hasattr(file, "uploaded_in_session") or not file.uploaded_in_session:
            df = read_csv_file(file)
            if df is not None:
                st.session_state["uploaded_dfs"].append((file.name.split('.')[0], df))
            file.uploaded_in_session = True

# --- بیمه برای هر دارایی (Married Put) ---
all_asset_names = [t for t, _ in st.session_state["downloaded_dfs"]] + [t for t, _ in st.session_state["uploaded_dfs"]]
for name in all_asset_names:
    with st.sidebar.expander(f"⚙️ بیمه برای {name}", expanded=False):
        st.markdown("""
        <div dir="rtl" style="text-align: right;">
        <b>Married Put چیست؟</b>
        <br>بیمه در پرتفو یعنی خرید همزمان دارایی و یک قرارداد اختیار فروش. اگر بیمه بگیرید، ضرر شما محدود می‌شود.
        </div>
        """, unsafe_allow_html=True)
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

# --- تحلیل پرتفو ---
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

    # --- تعدیل قیمت‌ها بر اساس بیمه ---
    for name in asset_names:
        if name in st.session_state["insured_assets"]:
            info = st.session_state["insured_assets"][name]
            insured = prices_df[name].copy()
            insured[insured < info['strike']] = info['strike']
            insured.iloc[0] -= info['premium']
            prices_df[name] = insured

    try:
        resampled_prices = prices_df.resample(resample_rule).last().dropna()
        returns = resampled_prices.pct_change().dropna()
        mean_returns = returns.mean() * annual_factor
        cov_matrix = returns.cov() * annual_factor
        std_devs = np.sqrt(np.diag(cov_matrix))

        # --- بهینه‌سازی پرتفو مونت‌کارلو و CVaR ---
        n_portfolios = 3000
        n_mc = 1000
        results = np.zeros((5 + len(asset_names), n_portfolios))
        cvar_results = np.zeros((3 + len(asset_names), n_portfolios))
        np.random.seed(42)
        rf = 0
        downside = returns.copy()
        downside[downside > 0] = 0
        for i in range(n_portfolios):
            weights = np.random.random(len(asset_names))
            weights /= np.sum(weights)
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            downside_risk = np.sqrt(np.dot(weights.T, np.dot(downside.cov() * annual_factor, weights)))
            sharpe_ratio = (port_return - rf) / port_std
            sortino_ratio = (port_return - rf) / downside_risk if downside_risk > 0 else np.nan
            mc_sims = np.random.multivariate_normal(mean_returns, cov_matrix, n_mc)
            port_mc_returns = np.dot(mc_sims, weights)
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
        show_periodic_risk_return(resampled_prices, best_weights, "پرتفو بهینه مونت‌کارلو")
        show_periodic_risk_return(resampled_prices, best_cvar_weights, f"پرتفو بهینه CVaR ({int(cvar_alpha*100)}%)")

        # --- نمودارهای Pie ---
        fig_pie = go.Figure(data=[
            go.Pie(labels=asset_names, values=best_weights * 100, hole=.5, textinfo='label+percent')
        ])
        fig_pie.update_layout(title="توزیع وزنی پرتفو بهینه (Monte Carlo)")
        st.plotly_chart(fig_pie, use_container_width=True)
        fig_pie_cvar = go.Figure(data=[
            go.Pie(labels=asset_names, values=best_cvar_weights * 100, hole=.5, textinfo='label+percent')
        ])
        fig_pie_cvar.update_layout(title=f"توزیع وزنی پرتفو بهینه بر اساس CVaR ({int(cvar_alpha*100)}%)")
        st.plotly_chart(fig_pie_cvar, use_container_width=True)

        # --- نمودار ریسک-بازده ---
        fig_mc = go.Figure()
        fig_mc.add_trace(
            go.Scatter(
                x=results[1]*100, y=results[0]*100,
                mode='markers',
                marker=dict(size=6, color=results[2], colorscale='Viridis', showscale=True, colorbar=dict(title='Sharpe Ratio')),
                name='پرتفوها'
            )
        )
        fig_mc.add_trace(go.Scatter(
            x=[results[1,best_idx]*100], y=[results[0,best_idx]*100],
            mode='markers+text',
            marker=dict(size=15, color='red'),
            text=["بهینه"], textposition="top right",
            name='پرتفوی بهینه'
        ))
        fig_mc.update_layout(title="نمودار ریسک-بازده پرتفوهای Monte Carlo", xaxis_title="ریسک (%)", yaxis_title="بازده (%)")
        st.plotly_chart(fig_mc, use_container_width=True)

        # --- مرز کارا ---
        ef_results, ef_weights = efficient_frontier(mean_returns, cov_matrix, annual_factor, points=200)
        max_sharpe_idx = np.argmax(ef_results[2])
        mpt_weights = ef_weights[max_sharpe_idx]
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=ef_results[0]*100, y=ef_results[1]*100,
            mode='markers', marker=dict(color=ef_results[2], colorscale='Viridis', size=7, showscale=True),
            name='مرز کارا'
        ))
        fig_ef.add_trace(go.Scatter(
            x=[ef_results[0, max_sharpe_idx]*100], y=[ef_results[1, max_sharpe_idx]*100],
            mode='markers+text', marker=dict(size=14, color='red', symbol='star'),
            text=["پرتفوی بهینه MPT"], textposition="top right",
            name='پرتفوی MPT'
        ))
        st.plotly_chart(fig_ef, use_container_width=True)

        # --- Max Drawdown ---
        st.subheader("🔻 بیشینه افت سرمایه (Max Drawdown) پرتفو")
        for label, w in [
            ("پرتفو بهینه مونت‌کارلو", best_weights),
            (f"پرتفو بهینه CVaR ({int(cvar_alpha*100)}%)", best_cvar_weights),
            ("پرتفو بهینه MPT", mpt_weights),
        ]:
            pf_prices = (resampled_prices * w).sum(axis=1)
            max_dd = calculate_max_drawdown(pf_prices)
            st.markdown(f"**{label}:** {max_dd:.2%}")

    except Exception as e:
        st.error(f"خطای تحلیل پرتفو: {e}")
else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price یا Close یا Open را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
