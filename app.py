import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

# ========= Session state =========
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "insured_assets" not in st.session_state:
    st.session_state["insured_assets"] = {}

# ========= File uploader =========
st.sidebar.header("بارگذاری فایل‌های قیمت دارایی (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "هر دارایی یک فایل CSV با ستون‌های Date و Price", type=['csv'], accept_multiple_files=True
)
if uploaded_files:
    st.session_state["uploaded_dfs"] = []
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            if "Date" not in df.columns or not any(col in df.columns for col in ["Price", "Close"]):
                st.warning(f"ستون‌های Date و Price/Close در {file.name} یافت نشد.")
                continue
            price_col = "Price" if "Price" in df.columns else "Close"
            df = df[["Date", price_col]].dropna()
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            df.columns = ["Price"]
            st.session_state["uploaded_dfs"].append((file.name.replace(".csv", ""), df))
        except Exception as e:
            st.warning(f"خطا در خواندن {file.name}: {e}")

# ========= Yahoo Finance =========
st.sidebar.markdown("---")
st.sidebar.subheader("دریافت مستقیم داده از Yahoo Finance")
ticker_input = st.sidebar.text_input("نماد (مثلاً BTC-USD یا AAPL)", "")
download_btn = st.sidebar.button("دریافت داده")
if download_btn and ticker_input:
    try:
        data = yf.download(ticker_input, period="max")
        if not data.empty:
            df = data.reset_index()[["Date", "Close"]].rename(columns={"Close": "Price"})
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
            st.session_state["downloaded_dfs"].append((ticker_input, df))
            st.success(f"داده {ticker_input} با موفقیت افزوده شد.")
        else:
            st.warning("داده‌ای برای این نماد یافت نشد.")
    except Exception as e:
        st.warning(f"خطا در دریافت داده: {e}")

# ========= Insurance & weights =========
all_asset_names = [t for t, _ in st.session_state["downloaded_dfs"]] + [t for t, _ in st.session_state["uploaded_dfs"]]
insured_assets = {}
st.sidebar.markdown("---")
st.sidebar.subheader("بیمه دارایی‌ها")
for name in all_asset_names:
    insured = st.sidebar.checkbox(f"فعال‌سازی بیمه برای {name}", key=f"insured_{name}")
    if insured:
        insured_assets[name] = True
st.session_state["insured_assets"] = insured_assets

st.sidebar.markdown("---")
st.sidebar.subheader("محدودیت وزن")
min_weights = {}
max_weights = {}
for name in all_asset_names:
    min_weights[name] = st.sidebar.number_input(f"حداقل وزن {name} (%)", 0.0, 100.0, 0.0, 1.0) / 100
    max_weights[name] = st.sidebar.number_input(f"حداکثر وزن {name} (%)", 0.0, 100.0, 100.0, 1.0) / 100

# ========= توضیحات ابزار =========
st.markdown("""
<div dir="rtl" style="text-align: right;">
<h3>ابزار تحلیل پرتفو (MPT + Monte Carlo + بیمه):</h3>
آپلود یا دانلود داده، بیمه دارایی، محدودیت وزن، پرتفوی بهینه مرز کارا و مونت‌کارلو و CVaR، خروجی وزن و ریسک و نمودارها.
</div>
""", unsafe_allow_html=True)

# ========= پردازش دیتافریم قیمت =========
all_dfs = st.session_state["downloaded_dfs"] + st.session_state["uploaded_dfs"]
if all_dfs:
    prices_df = pd.DataFrame()
    asset_names = []
    for t, df in all_dfs:
        prices_df = df.rename(columns={"Price": t}) if prices_df.empty else prices_df.join(df.rename(columns={"Price": t}), how="inner")
        asset_names.append(t)
    prices_df = prices_df.dropna()
    st.subheader("📉 روند قیمت دارایی‌ها")
    st.line_chart(prices_df)

    # بازده و کوواریانس
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean() * 12
    cov_matrix = returns.cov() * 12

    # ========= اثر بیمه روی کوواریانس =========
    cov_adj = cov_matrix.copy()
    for idx, name in enumerate(asset_names):
        if name in st.session_state["insured_assets"]:
            cov_adj.iloc[idx, idx] *= 0.2
            cov_adj.iloc[idx, :] *= 0.5
            cov_adj.iloc[:, idx] *= 0.5
            cov_adj.iloc[idx, idx] *= 0.8

    # ========= Efficient Frontier =========
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

    # ========= مرز کارا (MPT) =========
    ef_results, ef_weights = efficient_frontier(
        mean_returns, cov_adj, points=300,
        min_weights=np.array([min_weights[n] for n in asset_names]),
        max_weights=np.array([max_weights[n] for n in asset_names])
    )
    max_sharpe_idx = np.argmax(ef_results[2])
    mpt_weights = ef_weights[max_sharpe_idx]
    pf_prices_mpt = (prices_df * mpt_weights).sum(axis=1)
    pf_returns_mpt = pf_prices_mpt.pct_change().dropna()
    mean_ann_mpt = pf_returns_mpt.mean() * 12
    risk_ann_mpt = pf_returns_mpt.std() * np.sqrt(12)

    # ========= پرتفو مونت‌کارلو و CVaR =========
    points = 5000
    num_assets = len(asset_names)
    mc_weights = []
    mc_return = []
    mc_risk = []
    cvar_list = []
    for _ in range(points):
        while True:
            w = np.random.dirichlet(np.ones(num_assets), size=1)[0]
            w = np.maximum(w, [min_weights[n] for n in asset_names])
            w = np.minimum(w, [max_weights[n] for n in asset_names])
            w /= w.sum()
            if np.all(w >= [min_weights[n] for n in asset_names]) and np.all(w <= [max_weights[n] for n in asset_names]):
                break
        port_ret = np.dot(w, mean_returns)
        port_std = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        pf = (returns * w).sum(axis=1)
        cvar = pf[pf <= np.percentile(pf, 5)].mean()
        mc_weights.append(w)
        mc_return.append(port_ret)
        mc_risk.append(port_std)
        cvar_list.append(cvar)
    mc_weights = np.array(mc_weights)
    mc_return = np.array(mc_return)
    mc_risk = np.array(mc_risk)
    cvar_list = np.array(cvar_list)

    # MC Portfolio
    mc_idx = np.argmax(mc_return / mc_risk)
    mc_best_weights = mc_weights[mc_idx]
    pf_prices_mc = (prices_df * mc_best_weights).sum(axis=1)
    pf_returns_mc = pf_prices_mc.pct_change().dropna()
    mean_ann_mc = pf_returns_mc.mean() * 12
    risk_ann_mc = pf_returns_mc.std() * np.sqrt(12)

    # CVaR Portfolio
    cvar_idx = np.argmax(cvar_list)
    cvar_best_weights = mc_weights[cvar_idx]
    pf_prices_cvar = (prices_df * cvar_best_weights).sum(axis=1)
    pf_returns_cvar = pf_prices_cvar.pct_change().dropna()
    mean_ann_cvar = pf_returns_cvar.mean() * 12
    risk_ann_cvar = pf_returns_cvar.std() * np.sqrt(12)

    # ========= داشبورد =========
    st.markdown("#### 📊 پرتفو بهینه مرز کارا (MPT)")
    st.markdown(f"""<div dir="rtl" style="text-align:right">
    <b>سالانه:</b> بازده: {mean_ann_mpt:.2%} | ریسک: {risk_ann_mpt:.2%}<br>
    <b>وزن پرتفو:</b> {' | '.join([f"{n}: {w:.2%}" for n, w in zip(asset_names, mpt_weights)])}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 📊 پرتفو بهینه مونت‌کارلو")
    st.markdown(f"""<div dir="rtl" style="text-align:right">
    <b>سالانه:</b> بازده: {mean_ann_mc:.2%} | ریسک: {risk_ann_mc:.2%}<br>
    <b>وزن پرتفو:</b> {' | '.join([f"{n}: {w:.2%}" for n, w in zip(asset_names, mc_best_weights)])}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### 📊 پرتفو بهینه CVaR (95%)")
    st.markdown(f"""<div dir="rtl" style="text-align:right">
    <b>سالانه:</b> بازده: {mean_ann_cvar:.2%} | ریسک: {risk_ann_cvar:.2%}<br>
    <b>وزن پرتفو:</b> {' | '.join([f"{n}: {w:.2%}" for n, w in zip(asset_names, cvar_best_weights)])}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ===== نمودارهای دایره‌ای وزن =====
    st.markdown("#### توزیع وزنی پرتفو بهینه مرز کارا (MPT)")
    fig_pie_mpt = px.pie(
        values=mpt_weights,
        names=asset_names,
        hole=0.5
    )
    st.plotly_chart(fig_pie_mpt)

    st.markdown("#### توزیع وزنی پرتفو بهینه مونت‌کارلو")
    fig_pie_mc = px.pie(
        values=mc_best_weights,
        names=asset_names,
        hole=0.5
    )
    st.plotly_chart(fig_pie_mc)

    st.markdown("#### توزیع وزنی پرتفو بهینه CVaR")
    fig_pie_cvar = px.pie(
        values=cvar_best_weights,
        names=asset_names,
        hole=0.5
    )
    st.plotly_chart(fig_pie_cvar)

    # ===== نمودار مرز کارا =====
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
    st.markdown("#### نمودار مرز کارا و نقطه بهینه")
    st.plotly_chart(fig_all, use_container_width=True)

else:
    st.warning("لطفاً حداقل یک فایل CSV معتبر آپلود کنید یا داده‌ای از یاهو فایننس دریافت کنید.")
