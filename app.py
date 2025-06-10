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
# 7. Efficient Frontier با محدودیت وزن و اثر بیمه
# =========================
def efficient_frontier_with_ins(mean_returns, cov_matrix, annual_factor, points=200, min_weights=None, max_weights=None,
                               insured_assets=None, resampled_prices=None, asset_names=None):
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
        # اثر بیمه: اگر دارایی بیمه شده، واریانس آن را کاهش بده
        mean_mod = mean_returns.copy()
        cov_mod = cov_matrix.copy()
        if insured_assets and resampled_prices is not None and asset_names is not None:
            for idx, name in enumerate(asset_names):
                if name in insured_assets:
                    # واریانس را کاهش بده (مثلاً نصف کن یا حتی کمتر)
                    cov_mod.iloc[idx, idx] *= 0.25
                    # همبستگی با دیگران را کم کن
                    cov_mod.iloc[idx, :] *= 0.6
                    cov_mod.iloc[:, idx] *= 0.6
                    cov_mod.iloc[idx, idx] *= 0.7
        port_return = np.dot(weights, mean_mod)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_mod, weights)))
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

# ... بقیه بخش‌های بارگذاری داده، بیمه، محدودیت وزن و ... مثل قبل ...

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
    st.line_chart(prices_df.resample('M').last().dropna())

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    try:
        # 1. داده‌های اصلی
        resampled_prices = prices_df.resample('M').last().dropna()
        returns = resampled_prices.pct_change().dropna()
        mean_returns = returns.mean() * 12
        cov_matrix = returns.cov() * 12

        # 2. شبیه‌سازی مونت‌کارلو و CVaR (مانند قبل)
        annual_factor = 12

        # ====== مرز کارا با اثر بیمه ======
        ef_results, ef_weights = efficient_frontier_with_ins(mean_returns, cov_matrix, annual_factor, points=200,
            min_weights=np.array([0.0]*len(asset_names)),
            max_weights=np.array([1.0]*len(asset_names)),
            insured_assets=st.session_state["insured_assets"],
            resampled_prices=resampled_prices,
            asset_names=asset_names
        )
        max_sharpe_idx_ef = np.argmax(ef_results[2])
        mpt_weights = ef_weights[max_sharpe_idx_ef]
        # محاسبه بازده و ریسک پرتفو MPT
        pf_prices_mpt = (resampled_prices * mpt_weights).sum(axis=1)
        pf_returns_mpt = pf_prices_mpt.pct_change().dropna()
        mean_ann_mpt = pf_returns_mpt.mean() * annual_factor
        risk_ann_mpt = pf_returns_mpt.std() * (annual_factor ** 0.5)

        st.subheader("📊 پرتفو بهینه مرز کارا (MPT)")
        st.markdown(
            f"""<div dir="rtl" style="text-align:right">
            <b>سالانه:</b> بازده: {mean_ann_mpt:.2%} | ریسک: {risk_ann_mpt:.2%}<br>
            <b>وزن پرتفو:</b> {' | '.join([f"{n}: {w:.2%}" for n, w in zip(asset_names, mpt_weights)])}
            </div>
            """, unsafe_allow_html=True)

        # ====== نمودار مرز کارا ترکیبی ======
        fig_all = go.Figure()
        # مرز کارا
        fig_all.add_trace(go.Scatter(
            x=ef_results[0]*100, y=ef_results[1]*100,
            mode='lines+markers', marker=dict(color='gray', size=5), name='مرز کارا (MPT)'
        ))
        # نقطه بهینه MPT
        fig_all.add_trace(go.Scatter(
            x=[ef_results[0, max_sharpe_idx_ef]*100], y=[ef_results[1, max_sharpe_idx_ef]*100],
            mode='markers+text', marker=dict(size=16, color='red', symbol='star'),
            text=["MPT"], textposition="bottom right", name='پرتفو بهینه MPT'
        ))
        # (اختیاری) اگر پرتفوهای MC و CVaR داشتی، اینجا با همان وزن‌ها اضافه کن
        # فرض: best_weights و best_cvar_weights قبلا از مونت‌کارلو و CVaR به دست آمده‌اند
        # اگر آن‌ها را هم می‌خواهی نمایش بدهی:
        # fig_all.add_trace(...)  # مشابه توضیحات قبلی

        fig_all.update_layout(
            title="مرز کارا با تاثیر بیمه و نمایش پرتفوهای منتخب",
            xaxis_title="ریسک (%)",
            yaxis_title="بازده (%)"
        )
        st.plotly_chart(fig_all, use_container_width=True)

    except Exception as e:
        st.error(f"خطای تحلیل پرتفو: {e}")

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price یا Close یا Open را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
