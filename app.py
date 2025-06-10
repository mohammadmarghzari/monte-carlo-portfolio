import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import base64
from collections import Counter

# ========== Helper Functions ==========

# ... (تمام توابع کمکی شما بدون تغییر)

# اینجا همان توابع قبلی شما قرار می‌گیرند (format_money, format_percent, format_float, read_csv_file, get_price_dataframe_from_yf, calculate_max_drawdown, efficient_frontier, portfolio_risk_return, download_link)
# برای اختصار در اینجا حذف شده‌اند اما در فایل کامل باشند.

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

# ... (بخش حذف دارایی‌های دانلود و آپلود شده بدون تغییر)

# ========== Sidebar: Params and Yahoo Download ==========

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]

# نرخ بدون ریسک
st.sidebar.markdown("---")
st.sidebar.markdown("<b>نرخ بدون ریسک (برای نسبت شارپ و خط CML):</b>", unsafe_allow_html=True)
user_rf = st.sidebar.number_input("نرخ بدون ریسک سالانه (%)", min_value=0.0, max_value=100.0, value=3.0, step=0.1) / 100

user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, 0.25, 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

# ... (کد دانلود داده آنلاین یاهو فایننس و آپلود فایل بدون تغییر)

# ========== Sidebar: Investment Amount & Insurance ==========

# ... (مقدار سرمایه و بیمه هر دارایی بدون تغییر)

# ========== Main Analysis ==========

if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
    # ساخت دیتافریم قیمت با نام یکتا برای هر دارایی
    all_dfs = st.session_state["downloaded_dfs"] + st.session_state["uploaded_dfs"]
    name_counter = Counter()
    df_list = []
    asset_names = []
    for t, df in all_dfs:
        base_name = t
        name_counter[base_name] += 1
        name = base_name if name_counter[base_name] == 1 else f"{base_name} ({name_counter[base_name]})"
        temp_df = df.copy()
        temp_df = temp_df.rename(columns={"Price": name})
        asset_names.append(name)
        df_list.append(temp_df[[name]])
    if len(df_list) == 0:
        st.error("هیچ دیتافریمی برای پردازش وجود ندارد.")
        st.stop()
    prices_df = pd.concat(df_list, axis=1, join="inner")

    # --- نمایش روند قیمت
    with st.expander("📈 مشاهده روند قیمت دارایی‌ها", expanded=True):
        st.markdown("""
        <div dir="rtl" style="text-align:right">
        این نمودار روند قیمتی دارایی‌های شما در بازه انتخابی را به تفکیک نمایش می‌دهد. 
        داده‌ها بر اساس آخرین قیمت هر بازه (ماهانه/سه‌ماهه/شش‌ماهه) نمونه‌برداری شده‌اند.
        </div>
        """, unsafe_allow_html=True)
        st.line_chart(prices_df.resample(resample_rule).last().dropna())

    # --- تحلیل بازده و کوواریانس
    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = np.atleast_1d(np.array(returns.mean() * annual_factor))
    cov_matrix = np.atleast_2d(np.array(returns.cov() * annual_factor))
    std_devs = np.atleast_1d(np.sqrt(np.diag(cov_matrix)))

    # --- اثر بیمه روی کوواریانس (فقط دارایی‌های بیمه شده) و وزن‌دهی
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

    # --- شبیه‌سازی پرتفوها (MC، CVaR) با نرخ بدون ریسک
    n_portfolios = 3000
    n_mc = 1000
    results = np.zeros((5 + len(asset_names), n_portfolios))
    cvar_results = np.zeros((3 + len(asset_names), n_portfolios))
    np.random.seed(42)
    rf = user_rf  # نرخ بدون ریسک جدید

    downside = returns.copy()
    downside[downside > 0] = 0

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

    # --- مرز کارا با adjusted_cov و نرخ بدون ریسک
    ef_results, ef_weights = efficient_frontier(mean_returns, adjusted_cov, annual_factor, points=400)
    max_sharpe_idx = np.argmax((ef_results[1] - rf) / ef_results[0])  # Sharpe با نرخ بدون ریسک
    mpt_weights = ef_weights[max_sharpe_idx]

    # --- راهنمای سبک ها
    st.markdown("""
    <div dir="rtl" style="text-align:right">
    <b>راهنما:</b>
    <ul>
      <li><b>پرتفو بهینه مونت‌کارلو:</b> پرتفو با بهترین نسبت شارپ از بین پرتفوهای تصادفی.</li>
      <li><b>پرتفو بهینه CVaR:</b> پرتفو با کمترین ارزش در معرض ریسک شرطی (CVaR).</li>
      <li><b>پرتفو بهینه MPT:</b> پرتفو با بهترین نسبت شارپ روی مرز کارای نظری.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # --- داشبورد و نمودارها برای هر سبک
    styles = [
        ("پرتفو بهینه مونت‌کارلو", best_weights, "MC", "red"),
        (f"پرتفو بهینه CVaR ({int(cvar_alpha*100)}%)", best_cvar_weights, "CVaR", "green"),
        ("پرتفو بهینه MPT", mpt_weights, "MPT", "blue")
    ]
    style_points = {}

    for label, weights, code, color in styles:
        mean_m, risk_m, mean_a, risk_a = portfolio_risk_return(resampled_prices, weights, freq_label="M")
        # نقطه روی مرز کارا
        sharpe = (mean_a - rf) / risk_a
        style_points[code] = (risk_a, mean_a, sharpe)
        # داشبورد
        with st.expander(f"📋 {label} - راهنما و تحلیل", expanded=True):
            st.markdown(f"""
            <div dir="rtl" style="text-align:right">
            <b>{label}</b><br>
            این سبک بر اساس معیارهای بهینه‌سازی مختلف (شارپ، CVaR یا مرز کارا) انتخاب شده است.<br>
            وزن هر دارایی و ریسک و بازده پرتفو در این بخش نمایش داده شده است.
            </div>
            """, unsafe_allow_html=True)
            st.markdown(
                f"<b>سالانه:</b> بازده: {format_percent(mean_a)} | ریسک: {format_percent(risk_a)}<br>"
                f"<b>ماهانه:</b> بازده: {format_percent(mean_m)} | ریسک: {format_percent(risk_m)}",
                unsafe_allow_html=True
            )
            # نمودار دایره‌ای وزن‌دهی
            fig_pie = px.pie(
                values=weights,
                names=asset_names,
                title=f"توزیع وزنی پرتفو - {label}",
                hole=0.5
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # نمودار مرز کارا با خط CML و نقطه پرتفو
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ef_results[0], y=ef_results[1],
                mode="lines", line=dict(color="gray", width=2), name="مرز کارا"
            ))
            # خط CML
            max_risk = max(ef_results[0].max(), risk_a) * 1.1
            x_cml = np.linspace(0, max_risk, 100)
            y_cml = rf + sharpe * x_cml
            fig.add_trace(go.Scatter(
                x=x_cml, y=y_cml, mode="lines", line=dict(dash="dash", color=color), name="خط بازار سرمایه (CML)"
            ))
            # نقطه پرتفو
            fig.add_trace(go.Scatter(
                x=[risk_a], y=[mean_a], mode="markers+text", marker=dict(size=14, color=color, symbol="star"),
                text=[code], textposition="top right", name=f"{label}"
            ))
            fig.update_layout(
                title=f"مرز کارا و خط بازار سرمایه ({label})",
                xaxis_title="ریسک سالانه (انحراف معیار)",
                yaxis_title="بازده سالانه",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- نمودار مرز کارا نهایی با نمایش هر سه سبک پرتفو
    st.markdown("### 📈 مقایسه نهایی: مرز کارا و جایگاه پرتفوهای بهینه")
    st.markdown("""
    <div dir="rtl" style="text-align:right">
    این نمودار مرز کارای نظری را به همراه سه پرتفو بهینه (مونت‌کارلو، CVaR، MPT) و نسبت شارپ هرکدام نمایش می‌دهد.<br>
    نقاط هر سبک با رنگ متفاوت مشخص شده‌اند.
    </div>
    """, unsafe_allow_html=True)
    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=ef_results[0], y=ef_results[1],
        mode='lines', marker=dict(color='gray', size=5), name='مرز کارا'
    ))
    color_map = {"MC": "red", "CVaR": "green", "MPT": "blue"}
    for code, (risk, mean, sharpe) in style_points.items():
        fig_all.add_trace(go.Scatter(
            x=[risk], y=[mean], mode="markers+text",
            marker=dict(size=16, color=color_map[code], symbol="star"),
            text=[code], textposition="top right", name=f"پرتفو {code} (شارپ={sharpe:.2f})"
        ))
    fig_all.update_layout(
        title="مرز کارا با نمایش پرتفوهای بهینه سه سبک",
        xaxis_title="ریسک سالانه (انحراف معیار)",
        yaxis_title="بازده سالانه"
    )
    st.plotly_chart(fig_all, use_container_width=True)

    # --- بیمه (همان کد قبلی شما)
    # ... (کد بیمه و نمودار Married Put بدون تغییر)

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price یا Close یا Open را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
