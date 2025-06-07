import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import base64
from io import StringIO

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو، CVaR و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو، CVaR و استراتژی Married Put")

# ------------------- SESSION STATE INITIALIZATION -------------------
if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "insured_assets" not in st.session_state:
    st.session_state["insured_assets"] = {}

# ------------------- UTILITY FUNCTIONS -------------------
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

def download_link(df, filename):
    csv = df.reset_index(drop=True).to_csv(index=False).encode()
    b64 = base64.b64encode(csv).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ دریافت فایل CSV</a>'

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

# ------------------- SIDEBAR UI -------------------
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True, key="uploader"
)

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, 0.25, 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

with st.sidebar.expander("📥 دانلود داده آنلاین از یاهو فاینانس"):
    st.markdown("""
    <div dir="rtl" style="text-align: right; font-size: 14px">
    <b>راهنما:</b><br>
    - نماد هر دارایی را مطابق سایت یاهو فاینانس وارد کنید.<br>
    - نمادها را با کاما و بدون فاصله وارد کنید.<br>
    - مثال: <b>BTC-USD,AAPL,GOOGL,ETH-USD</b><br>
    - برای بیت‌کوین: <b>BTC-USD</b> <br>
    - برای اپل: <b>AAPL</b> <br>
    - برای اتریوم: <b>ETH-USD</b> <br>
    - برای شاخص S&P500: <b>^GSPC</b> <br>
    </div>
    """, unsafe_allow_html=True)
    tickers_input = st.text_input("نماد دارایی‌ها (با کاما و بدون فاصله)")
    start = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
    end = st.date_input("تاریخ پایان", value=pd.to_datetime("today"))
    download_btn = st.button("دریافت داده آنلاین")

# ------------------- YAHOO FINANCE DOWNLOAD ACTION -------------------
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
            # اضافه کردن به لیست دائمی session_state
            st.session_state["downloaded_dfs"].extend(new_downloaded)
    except Exception as ex:
        st.error(f"خطا در دریافت داده: {ex}")

# ------------------- SHOW ADDED DATA -------------------
if st.session_state["downloaded_dfs"]:
    st.markdown('<div dir="rtl" style="text-align: right;"><b>داده‌های دانلودشده از یاهو فاینانس:</b></div>', unsafe_allow_html=True)
    for t, df in st.session_state["downloaded_dfs"]:
        st.markdown(f"<div dir='rtl' style='text-align: right;'><b>{t}</b></div>", unsafe_allow_html=True)
        st.dataframe(df.head())

# ------------------- FILE UPLOAD ACTION -------------------
if uploaded_files:
    for file in uploaded_files:
        if not hasattr(file, "uploaded_in_session") or not file.uploaded_in_session:
            df = read_csv_file(file)
            if df is not None:
                st.session_state["uploaded_dfs"].append((file.name.split('.')[0], df))
            file.uploaded_in_session = True

# ------------------- بیمه برای همه دارایی‌ها -------------------
all_asset_names = [t for t, _ in st.session_state["downloaded_dfs"]] + [t for t, _ in st.session_state["uploaded_dfs"]]
for name in all_asset_names:
    with st.sidebar.expander(f"⚙️ بیمه برای {name}", expanded=False):
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

# ------------------- BUILD FINAL PRICE DATAFRAME -------------------
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

    st.subheader("🧪 پیش‌نمایش داده‌ی نهایی برای تحلیل پرتفو")
    st.write(prices_df.head())
    st.write("شکل داده:", prices_df.shape)

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    try:
        resampled_prices = prices_df.resample(resample_rule).last().dropna()
        returns = resampled_prices.pct_change().dropna()
        mean_returns = returns.mean() * annual_factor
        cov_matrix = returns.cov() * annual_factor
        std_devs = np.sqrt(np.diag(cov_matrix))

        st.subheader("📉 اولین نمودار قیمت دارایی‌ها")
        st.line_chart(resampled_prices)

        # تحلیل پرتفو با مونت‌کارلو و CVaR
        n_portfolios = 3000
        n_mc = 1000
        results = np.zeros((5 + len(asset_names), n_portfolios))
        np.random.seed(42)
        rf = 0
        downside = returns.copy()
        downside[downside > 0] = 0

        adjusted_cov = cov_matrix.copy()
        preference_weights = []
        for i, name in enumerate(asset_names):
            if name in st.session_state["insured_assets"]:
                risk_scale = 1 - st.session_state["insured_assets"][name]['loss_percent'] / 100
                adjusted_cov.iloc[i, :] *= risk_scale
                adjusted_cov.iloc[:, i] *= risk_scale
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

        best_idx = np.argmin(np.abs(results[1] - user_risk))
        best_return = results[0, best_idx]
        best_risk = results[1, best_idx]
        best_sharpe = results[2, best_idx]
        best_sortino = results[3, best_idx]
        best_weights = results[5:, best_idx]

        best_cvar_idx = np.argmin(results[4])
        best_cvar_return = results[0, best_cvar_idx]
        best_cvar_risk = results[1, best_cvar_idx]
        best_cvar_cvar = results[4, best_cvar_idx]
        best_cvar_weights = results[5:, best_cvar_idx]

        st.subheader("📊 داشبورد خلاصه پرتفو")
        st.markdown(f'''
        <div dir="rtl" style="text-align: right">
        <b>بازده سالانه پرتفو:</b> {best_return:.2%}<br>
        <b>ریسک سالانه پرتفو:</b> {best_risk:.2%}<br>
        <b>نسبت شارپ:</b> {best_sharpe:.2f}<br>
        <b>بیشترین وزن:</b> {asset_names[np.argmax(best_weights)]} ({np.max(best_weights)*100:.2f}%)<br>
        <b>کمترین وزن:</b> {asset_names[np.argmin(best_weights)]} ({np.min(best_weights)*100:.2f}%)<br>
        </div>
        ''', unsafe_allow_html=True)
        fig_pie = go.Figure(data=[go.Pie(labels=asset_names, values=best_weights * 100, hole=.5, textinfo='label+percent')])
        fig_pie.update_layout(title="توزیع وزنی پرتفو بهینه")
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("📈 پرتفو بهینه (مونت‌کارلو)")
        st.markdown(f"""
        - ✅ بازده سالانه: **{best_return:.2%}**
        - ⚠️ ریسک سالانه (انحراف معیار): **{best_risk:.2%}**
        - 🧠 نسبت شارپ: **{best_sharpe:.2f}**
        - 📉 نسبت سورتینو: **{best_sortino:.2f}**
        """)
        for i, name in enumerate(asset_names):
            st.markdown(f"🔹 وزن {name}: {best_weights[i]*100:.2f}%")

        st.subheader(f"🟢 پرتفو بهینه بر اساس CVaR ({int(cvar_alpha*100)}%)")
        st.markdown(f"""
        - ✅ بازده سالانه: **{best_cvar_return:.2%}**
        - ⚠️ ریسک سالانه (انحراف معیار): **{best_cvar_risk:.2%}**
        - 🟠 CVaR ({int(cvar_alpha*100)}%): **{best_cvar_cvar:.2%}**
        """)
        for i, name in enumerate(asset_names):
            st.markdown(f"🔸 وزن {name}: {best_cvar_weights[i]*100:.2f}%")

        st.subheader("🌈 مرز کارا (Efficient Frontier) با روش MPT")
        ef_results, ef_weights = efficient_frontier(mean_returns, cov_matrix, annual_factor, points=200)
        max_sharpe_idx = np.argmax(ef_results[2])
        mpt_return = ef_results[1, max_sharpe_idx]
        mpt_risk = ef_results[0, max_sharpe_idx]
        mpt_weights = ef_weights[max_sharpe_idx]
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(
            x=ef_results[0]*100, y=ef_results[1]*100,
            mode='markers', marker=dict(color=ef_results[2], colorscale='Viridis', size=7, showscale=True),
            name='مرز کارا'
        ))
        fig_ef.add_trace(go.Scatter(
            x=[mpt_risk*100], y=[mpt_return*100],
            mode='markers+text', marker=dict(size=14, color='red', symbol='star'),
            text=["پرتفوی بهینه MPT"], textposition="top right",
            name='پرتفوی MPT'
        ))
        st.plotly_chart(fig_ef, use_container_width=True)

        st.subheader("🔻 بیشینه افت سرمایه (Max Drawdown) پرتفو")
        for label, w in [
            ("پرتفو بهینه مونت‌کارلو", best_weights),
            (f"پرتفو بهینه CVaR ({int(cvar_alpha*100)}%)", best_cvar_weights),
            ("پرتفو بهینه MPT", mpt_weights),
        ]:
            pf_prices = (resampled_prices * w).sum(axis=1)
            max_dd = calculate_max_drawdown(pf_prices)
            st.markdown(f"**{label}:** {max_dd:.2%}")

        st.subheader("📉 بیمه دارایی‌ها (Married Put)")
        for name in st.session_state["insured_assets"]:
            info = st.session_state["insured_assets"][name]
            x = np.linspace(info['spot'] * 0.5, info['spot'] * 1.5, 200)
            asset_pnl = (x - info['spot']) * info['base']
            put_pnl = np.where(x < info['strike'], (info['strike'] - x) * info['amount'], 0) - info['premium'] * info['amount']
            total_pnl = asset_pnl + put_pnl
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=x[total_pnl>=0], y=total_pnl[total_pnl>=0], mode='lines', name='سود', line=dict(color='green', width=3)
            ))
            fig2.add_trace(go.Scatter(
                x=x[total_pnl<0], y=total_pnl[total_pnl<0], mode='lines', name='زیان', line=dict(color='red', width=3)
            ))
            fig2.add_trace(go.Scatter(
                x=x, y=asset_pnl, mode='lines', name='دارایی پایه', line=dict(dash='dot', color='gray')
            ))
            fig2.add_trace(go.Scatter(
                x=x, y=put_pnl, mode='lines', name='پوت', line=dict(dash='dot', color='blue')
            ))
            st.markdown(f"**{name}**")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🔮 پیش‌بینی قیمت و بازده آتی هر دارایی")
        future_months = 6 if period == 'شش‌ماهه' else (3 if period == 'سه‌ماهه' else 1)
        for i, name in enumerate(asset_names):
            last_price = resampled_prices[name].iloc[-1]
            mu = mean_returns[i] / annual_factor
            sigma = std_devs[i] / np.sqrt(annual_factor)
            sim_prices = []
            n_sim = 500
            for _ in range(n_sim):
                sim = last_price * np.exp(np.cumsum(np.random.normal(mu, sigma, future_months)))
                sim_prices.append(sim[-1])
            sim_prices = np.array(sim_prices)
            future_price_mean = np.mean(sim_prices)
            future_return = (future_price_mean - last_price) / last_price

            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(x=sim_prices, nbinsx=20, name="پیش‌بینی قیمت", marker_color='purple'))
            fig3.add_vline(x=future_price_mean, line_dash="dash", line_color="green", annotation_text=f"میانگین: {future_price_mean:.2f}")
            fig3.update_layout(title=f"پیش‌بینی قیمت {name} در {future_months} {'ماه' if future_months>1 else 'ماه'} آینده",
                xaxis_title="قیمت انتهایی", yaxis_title="تعداد شبیه‌سازی")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(f"📈 **میانگین قیمت آینده:** {future_price_mean:.2f} | 📊 **درصد بازده آتی:** {future_return:.2%}")

    except Exception as e:
        st.error(f"خطای تحلیل پرتفو: {e}")

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price یا Close یا Open را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
