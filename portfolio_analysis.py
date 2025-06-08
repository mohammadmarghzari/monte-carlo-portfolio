import streamlit as st
import numpy as np
import pandas as pd
from ui_helpers import format_money, format_percent, format_float

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

def run_portfolio_analysis(resample_rule, annual_factor, user_risk, cvar_alpha):
    st.subheader("📉 روند قیمت دارایی‌ها")
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
    except Exception as e:
        st.error(f"خطای تحلیل پرتفو: {e}")
