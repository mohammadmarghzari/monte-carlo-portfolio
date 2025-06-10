import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================
# 1. Session State Init
# =========================
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "insured_assets" not in st.session_state:
    st.session_state["insured_assets"] = {}

# =========================
# 2. File Uploader
# =========================
st.sidebar.header("بارگذاری فایل‌های قیمت دارایی (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "هر دارایی یک فایل CSV با ستون‌های Date و Price", type=['csv'], accept_multiple_files=True
)
if uploaded_files:
    st.session_state["uploaded_dfs"] = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            if "Date" not in df.columns or "Price" not in df.columns:
                st.warning(f"ستون‌های Date و Price در {file.name} یافت نشد.")
                continue
            df = df[["Date", "Price"]].dropna()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            st.session_state["uploaded_dfs"].append((file.name.replace(".csv", ""), df))
        except Exception as e:
            st.warning(f"خطا در خواندن {file.name}: {e}")

# =========================
# 3. بیمه دارایی‌ها و محدودیت وزن
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("بیمه دارایی‌ها")
insured_assets = {}
asset_names = [t for t, _ in st.session_state["uploaded_dfs"]]
for name in asset_names:
    insured = st.sidebar.checkbox(f"فعال‌سازی بیمه برای {name}", key=f"insured_{name}")
    if insured:
        insured_assets[name] = True
st.session_state["insured_assets"] = insured_assets

st.sidebar.markdown("---")
st.sidebar.subheader("محدودیت وزن")
min_weights = {}
max_weights = {}
for name in asset_names:
    min_weights[name] = st.sidebar.number_input(f"حداقل وزن {name} (%)", 0.0, 100.0, 0.0, 1.0) / 100
    max_weights[name] = st.sidebar.number_input(f"حداکثر وزن {name} (%)", 0.0, 100.0, 100.0, 1.0) / 100

# =========================
# 4. توضیحات ابزار
# =========================
st.markdown("""
<div dir="rtl" style="text-align: right;">
<h3>ابزار تحلیل پرتفو: توضیحات کلی</h3>
این ابزار برای تحلیل و بهینه‌سازی ترکیب دارایی‌های یک پرتفو (Portfolio) با قابلیت‌های حرفه‌ای شبیه‌سازی مونت‌کارلو، مرز کارا (MPT)، بیمه و محدودیت‌های سفارشی طراحی شده است.
</div>
""", unsafe_allow_html=True)

# =========================
# 5. تحلیل پرتفو و محاسبه
# =========================
if st.session_state["uploaded_dfs"]:
    # ساخت دیتافریم قیمت‌ها
    prices_df = pd.DataFrame()
    for t, df in st.session_state["uploaded_dfs"]:
        prices_df = df.rename(columns={"Price": t}) if prices_df.empty else prices_df.join(df.rename(columns={"Price": t}), how="inner")
    prices_df = prices_df.dropna()
    asset_names = list(prices_df.columns)
    st.line_chart(prices_df)

    # محاسبه بازده‌ها و کوواریانس سالانه
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12

    # اثر بیمه روی کوواریانس
    cov_adj = cov_matrix.copy()
    for idx, name in enumerate(asset_names):
        if name in st.session_state["insured_assets"]:
            cov_adj.iloc[idx, idx] *= 0.2
            cov_adj.iloc[idx, :] *= 0.5
            cov_adj.iloc[:, idx] *= 0.5
            cov_adj.iloc[idx, idx] *= 0.8

    # تابع مرز کارا (Efficient Frontier)
    def efficient_frontier(mean_returns, cov_matrix, points=200, min_weights=None, max_weights=None):
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

    # محاسبه مرز کارا با اثر بیمه و محدودیت وزن
    ef_results, ef_weights = efficient_frontier(
        mean_returns, cov_adj, points=300,
        min_weights=np.array([min_weights[n] for n in asset_names]),
        max_weights=np.array([max_weights[n] for n in asset_names])
    )
    max_sharpe_idx = np.argmax(ef_results[2])
    mpt_weights = ef_weights[max_sharpe_idx]

    # ریسک/بازده پرتفو MPT
    pf_prices_mpt = (prices_df * mpt_weights).sum(axis=1)
    pf_returns_mpt = pf_prices_mpt.pct_change().dropna()
    mean_ann_mpt = pf_returns_mpt.mean() * 12
    risk_ann_mpt = pf_returns_mpt.std() * np.sqrt(12)

    # نمایش داشبورد پرتفو بهینه مرز کارا (MPT)
    st.markdown("#### 📊 پرتفو بهینه مرز کارا (MPT)")
    st.markdown(f"""<div dir="rtl" style="text-align:right">
    <b>سالانه:</b> بازده: {mean_ann_mpt:.2%} | ریسک: {risk_ann_mpt:.2%}<br>
    <b>وزن پرتفو:</b> {' | '.join([f"{n}: {w:.2%}" for n, w in zip(asset_names, mpt_weights)])}
    </div>
    """, unsafe_allow_html=True)
    # نمودار وزن‌دهی
    import plotly.express as px
    mpt_weight_series = pd.Series(mpt_weights, index=asset_names)
    fig_pie_mpt = px.pie(
        values=mpt_weight_series.values,
        names=mpt_weight_series.index,
        title="توزیع وزنی پرتفو بهینه مرز کارا (MPT)",
        hole=0.5
    )
    st.plotly_chart(fig_pie_mpt)

    # نمودار مرز کارا و نقطه بهینه
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=ef_results[0]*100, y=ef_results[1]*100,
        mode='lines+markers', marker=dict(color='gray', size=5), name='مرز کارا (MPT)'
    ))
    fig_all.add_trace(go.Scatter(
        x=[ef_results[0, max_sharpe_idx]*100], y=[ef_results[1, max_sharpe_idx]*100],
        mode='markers+text', marker=dict(size=16, color='red', symbol='star'),
        text=["MPT"], textposition="bottom right", name='پرتفو بهینه MPT'
    ))
    fig_all.update_layout(
        title="مرز کارا با تاثیر بیمه و نمایش پرتفو بهینه MPT",
        xaxis_title="ریسک (%)",
        yaxis_title="بازده (%)"
    )
    st.plotly_chart(fig_all, use_container_width=True)

else:
    st.warning("لطفاً حداقل یک فایل CSV معتبر آپلود کنید.")
