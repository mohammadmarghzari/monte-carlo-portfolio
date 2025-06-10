import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import base64

# ========== Helper Functions ==========

def format_money(val):
    """فرمت نمایش دلاری: 2572 دلار یا 0.123 دلار"""
    if val == 0:
        return "۰ دلار"
    elif val >= 1:
        return "{:,.0f} دلار".format(val)
    else:
        return "{:.3f} دلار".format(val).replace('.', '٫')

def format_percent(val):
    """فرمت درصد با سه اعشار و بدون صفر اضافی"""
    return "{:.3f}%".format(val*100).replace('.', '٫')

def format_float(val):
    """فرمت عدد اعشاری برای پارامترهای بیمه و ..."""
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

def calculate_max_drawdown(prices: pd.Series) -> float:
    roll_max = prices.cummax()
    drawdown = (prices - roll_max) / roll_max
    max_dd = drawdown.min()
    return max_dd

def efficient_frontier(mean_returns, cov_matrix, annual_factor, points=200):
    mean_returns = np.atleast_1d(np.array(mean_returns))
    cov_matrix = np.atleast_2d(np.array(cov_matrix))
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

def portfolio_risk_return(resampled_prices, weights, freq_label="M"):
    pf_prices = (resampled_prices * weights).sum(axis=1)
    pf_returns = pf_prices.pct_change().dropna()
    if freq_label == "M":
        ann_factor = 12
    elif freq_label == "W":
        ann_factor = 52
    else:
        ann_factor = 1
    mean_month = pf_returns.mean()
    risk_month = pf_returns.std()
    mean_ann = mean_month * ann_factor
    risk_ann = risk_month * (ann_factor ** 0.5)
    return mean_month, risk_month, mean_ann, risk_ann

def download_link(df, filename):
    csv = df.reset_index(drop=True).to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ دریافت فایل CSV</a>'

# ========== Session State ==========

if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "insured_assets" not in st.session_state:
    st.session_state["insured_assets"] = {}
if "investment_amount" not in st.session_state:
    st.session_state["investment_amount"] = 1000.0

# ========== Sidebar: File Upload/Delete ==========

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
                st.session_state["insured_assets"].pop(t, None)
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

all_asset_names = [t for t, _ in st.session_state["downloaded_dfs"]] + [t for t, _ in st.session_state["uploaded_dfs"]]

with st.sidebar.expander("💵 مقدار کل سرمایه‌گذاری (معادل دلاری)", expanded=True):
    investment_amount = st.text_input("مقدار سرمایه کل (دلار)", value=format_float(st.session_state["investment_amount"]).replace(',', ''), key="inv_amount_inp")
    try:
        val = float(investment_amount.replace('٫', '.').replace(',', ''))
        if val < 0:
            val = 0
    except:
        val = 0
    st.session_state["investment_amount"] = val
    st.markdown(f"مقدار واردشده: <b>{format_money(val)}</b>", unsafe_allow_html=True)

for name in all_asset_names:
    with st.sidebar.expander(f"⚙️ بیمه برای {name}", expanded=False):
        st.markdown("""
        <div dir="rtl" style="text-align: right;">
        <b>Married Put چیست؟</b>
        <br>بیمه در پرتفو (Married Put) یعنی شما همزمان با نگهداری دارایی، یک قرارداد اختیار فروش (Put Option) برای همان دارایی [...]
        </div>
        """, unsafe_allow_html=True)
        insured = st.checkbox(f"فعال‌سازی بیمه برای {name}", key=f"insured_{name}")
        if insured:
            loss_percent = st.number_input(f"📉 درصد ضرر معامله پوت برای {name}", 0.0, 100.0, 30.0, step=0.01, format="%.3f", key=f"loss_{name}")
            strike = st.number_input(f"🎯 قیمت اعمال پوت برای {name}", 0.0, 1e6, 100.0, step=0.01, format="%.3f", key=f"strike_{name}")
            premium = st.number_input(f"💰 قیمت قرارداد پوت برای {name}", 0.0, 1e6, 5.0, step=0.01, format="%.3f", key=f"premium_{name}")
            amount = st.number_input(f"📦 مقدار قرارداد برای {name}", 0.0, 1e6, 1.0, step=0.01, format="%.3f", key=f"amount_{name}")
            spot_price = st.number_input(f"📌 قیمت فعلی دارایی پایه {name}", 0.0, 1e6, 100.0, step=0.01, format="%.3f", key=f"spot_{name}")
            asset_amount = st.number_input(f"📦 مقدار دارایی پایه {name}", 0.0, 1e6, 1.0, step=0.01, format="%.3f", key=f"base_{name}")
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

    st.subheader("📉 روند قیمت دارایی‌ها")
    st.line_chart(prices_df.resample(resample_rule).last().dropna())

    if prices_df.empty or len(asset_names) < 1:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    if len(asset_names) < 2:
        st.warning('برای تحلیل پرتفوی، حداقل دو دارایی انتخاب کنید.')
        st.stop()

    try:
        resampled_prices = prices_df.resample(resample_rule).last().dropna()
        returns = resampled_prices.pct_change().dropna()
        mean_returns = np.atleast_1d(np.array(returns.mean() * annual_factor))
        cov_matrix = np.atleast_2d(np.array(returns.cov() * annual_factor))
        std_devs = np.atleast_1d(np.sqrt(np.diag(cov_matrix)))

        n_portfolios = 3000
        n_mc = 1000
        results = np.zeros((5 + len(asset_names), n_portfolios))
        cvar_results = np.zeros((3 + len(asset_names), n_portfolios))
        np.random.seed(42)
        rf = 0
        downside = returns.copy()
        downside[downside > 0] = 0
        adjusted_cov = cov_matrix.copy()
        preference_weights = []
        for i, name in enumerate(asset_names):
            if name in st.session_state["insured_assets"]:
                risk_scale = 1 - st.session_state["insured_assets"][name]['loss_percent'] / 100
                adjusted_cov[i, :] *= risk_scale
                adjusted_cov[:, i] *= risk_scale
                preference_weights.append(1 / (std_devs[i] * risk_scale**0.7))
            else:
                preference_weights.append(1 / std_devs[i])
        preference_weights = np.array(preference_weights)
        preference_weights /= np.sum(preference_weights)
        for i in range(n_portfolios):
            weights = np.random.random(len(asset_names)) * preference_weights
            weights /= np.sum(weights)
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))
            downside_risk = np.sqrt(np.dot(weights.T, np.dot(downside.cov() * annual_factor, weights)))
            sharpe_ratio = (port_return - rf) / port_std
            sortino_ratio = (port_return - rf) / downside_risk if downside_risk > 0 else np.nan
            mc_sims = np.random.multivariate_normal(mean_returns/annual_factor, adjusted_cov/annual_factor, n_mc)
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

        ef_results, ef_weights = efficient_frontier(mean_returns, cov_matrix, annual_factor, points=200)
        max_sharpe_idx = np.argmax(ef_results[2])
        mpt_weights = ef_weights[max_sharpe_idx]

        # نمایش هم درصد و هم مقدار دلاری هر دارایی برای هر سبک
        st.markdown("### 💰 ترکیب سرمایه‌گذاری هر سبک (درصد و دلار)")
        for label, weights in [
            ("پرتفو بهینه مونت‌کارلو", best_weights),
            (f"پرتفو بهینه CVaR ({int(cvar_alpha*100)}%)", best_cvar_weights),
            ("پرتفو بهینه MPT", mpt_weights)
        ]:
            st.markdown(f"**{label}:**")
            cols = st.columns(len(asset_names))
            for i, name in enumerate(asset_names):
                percent = weights[i]
                dollar = percent * st.session_state["investment_amount"]
                with cols[i]:
                    st.markdown(f"""
                    <div style='text-align:center;direction:rtl'>
                    <b>{name}</b><br>
                    {format_percent(percent)}<br>
                    {format_money(dollar)}
                    </div>
                    """, unsafe_allow_html=True)

        st.subheader("📊 داشبورد خلاصه پرتفو")
        for label, weights in [
            ("پرتفو بهینه مونت‌کارلو", best_weights),
            (f"پرتفو بهینه CVaR ({int(cvar_alpha*100)}%)", best_cvar_weights),
            ("پرتفو بهینه MPT", mpt_weights)
        ]:
            mean_m, risk_m, mean_a, risk_a = portfolio_risk_return(resampled_prices, weights, freq_label="M")
            st.markdown(
                f"<b>{label}</b> <br>"
                f"سالانه: بازده: {format_percent(mean_a)} | ریسک: {format_percent(risk_a)}<br>"
                f"ماهانه: بازده: {format_percent(mean_m)} | ریسک: {format_percent(risk_m)}",
                unsafe_allow_html=True
            )

        st.markdown("### 📋 جزئیات ریسک و بازده هر سبک")
        for label, weights in [
            ("پرتفو بهینه مونت‌کارلو", best_weights),
            (f"پرتفو بهینه CVaR ({int(cvar_alpha*100)}%)", best_cvar_weights),
            ("پرتفو بهینه MPT", mpt_weights)
        ]:
            mean_m, risk_m, mean_a, risk_a = portfolio_risk_return(resampled_prices, weights, freq_label="M")
            st.markdown(f"""<div dir="rtl" style="text-align:right">
            <b>{label}</b><br>
            • بازده ماهانه: {format_percent(mean_m)} <br>
            • ریسک ماهانه: {format_percent(risk_m)} <br>
            • بازده سالانه: {format_percent(mean_a)} <br>
            • ریسک سالانه: {format_percent(risk_a)}
            </div>""", unsafe_allow_html=True)

        # ... (تمام نمودارها و سایر تحلیل‌ها مثل قبل - حذف برای اختصار)
        # نمودار سود و زیان بیمه (Married Put) با نمایش درصد صحیح و نقطه سر به سر دقیق
        st.subheader("📉 بیمه دارایی‌ها (Married Put)")
        for name in st.session_state["insured_assets"]:
            info = st.session_state["insured_assets"][name]
            x = np.linspace(info['spot'] * 0.5, info['spot'] * 1.5, 400)
            asset_pnl = (x - info['spot']) * info['base']
            put_pnl = np.where(x < info['strike'], (info['strike'] - x) * info['amount'], 0) - info['premium'] * info['amount']
            total_pnl = asset_pnl + put_pnl

            initial_cost = info['spot'] * info['base'] + info['premium'] * info['amount']

            percent_profit = np.where(initial_cost != 0, 100 * total_pnl / initial_cost, 0)

            # نقطه سر به سر دقیق (جایی که total_pnl نزدیک صفر است)
            idx_be = np.argmin(np.abs(total_pnl))
            break_even = x[idx_be]
            break_even_y = total_pnl[idx_be]
            break_even_percent = percent_profit[idx_be]

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=x[total_pnl>=0],
                y=total_pnl[total_pnl>=0],
                mode='lines',
                name='سود',
                line=dict(color='green', width=3),
                customdata=np.stack([percent_profit[total_pnl>=0]], axis=-1),
                hovertemplate='قیمت: %{x:.3f}<br>سود: %{y:.3f}<br>درصد سود: %{customdata[0]:.2f}%<extra></extra>'
            ))
            fig2.add_trace(go.Scatter(
                x=x[total_pnl<0],
                y=total_pnl[total_pnl<0],
                mode='lines',
                name='زیان',
                line=dict(color='red', width=3),
                customdata=np.stack([percent_profit[total_pnl<0]], axis=-1),
                hovertemplate='قیمت: %{x:.3f}<br>زیان: %{y:.3f}<br>درصد زیان: %{customdata[0]:.2f}%<extra></extra>'
            ))
            fig2.add_trace(go.Scatter(
                x=x, y=asset_pnl, mode='lines', name='دارایی پایه', line=dict(dash='dot', color='gray')
            ))
            fig2.add_trace(go.Scatter(
                x=x, y=put_pnl, mode='lines', name='پوت', line=dict(dash='dot', color='blue')
            ))
            # نقطه سر به سر دقیق
            fig2.add_trace(go.Scatter(
                x=[break_even], y=[break_even_y], mode='markers+text',
                marker=dict(size=14, color='orange', symbol='x'),
                text=[f'سر به سر\n{break_even:.2f}\n{break_even_percent:.2f}%'],
                textposition="top right",
                name='نقطه سر به سر',
                hovertemplate='قیمت سر به سر: %{x:.3f}<br>بازده: %{y:.3f}<br>درصد: ' + f'{break_even_percent:.2f}%<extra></extra>'
            ))
            st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
            st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"خطای تحلیل پرتفو: {e}")

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price یا Close یا Open را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")