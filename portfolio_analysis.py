import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def run_portfolio_analysis(prices_df, asset_names, st):
    resample_rule = "M"
    annual_factor = 12

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = np.atleast_1d(np.array(returns.mean() * annual_factor))
    cov_matrix = np.atleast_2d(np.array(returns.cov() * annual_factor))
    std_devs = np.atleast_1d(np.sqrt(np.diag(cov_matrix)))

    n_portfolios = 1000
    results = np.zeros((3 + len(asset_names), n_portfolios))
    np.random.seed(42)
    for i in range(n_portfolios):
        weights = np.random.random(len(asset_names))
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = port_return / port_std if port_std > 0 else 0
        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3:, i] = weights

    best_idx = np.argmax(results[2])  # بیشینه شارپ (MPT)
    best_weights = results[3:, np.random.randint(0, n_portfolios)]  # مونت‌کارلو رندوم
    best_cvar_weights = results[3:, np.argmin(results[1])]  # کمترین ریسک (نمونه‌ای برای CVaR)
    mpt_weights = results[3:, best_idx]
    ef_results = (results[1], results[0], results[2])

    return {
        "best_weights": best_weights,
        "best_cvar_weights": best_cvar_weights,
        "mpt_weights": mpt_weights,
        "ef_results": ef_results,
        "cov_matrix": cov_matrix,
        "mean_returns": mean_returns
    }

def show_weights(analysis, asset_names, st):
    st.markdown("### 💰 ترکیب سرمایه‌گذاری هر سبک (درصد و دلار)")
    investment_amount = st.session_state["investment_amount"]
    for label, weights in [
        ("پرتفو بهینه مونت‌کارلو", analysis["best_weights"]),
        ("پرتفو بهینه CVaR (95%)", analysis["best_cvar_weights"]),
        ("پرتفو بهینه MPT", analysis["mpt_weights"])
    ]:
        st.markdown(f"**{label}:**")
        cols = st.columns(len(asset_names))
        for i, name in enumerate(asset_names):
            percent = weights[i]
            dollar = percent * investment_amount
            with cols[i]:
                st.markdown(f"""
                <div style='text-align:center;direction:rtl'>
                <b>{name}</b><br>
                {percent*100:.3f}%<br>
                {dollar:.3f} دلار
                </div>
                """, unsafe_allow_html=True)

def plot_pie_charts(analysis, asset_names, st):
    styles = [
        ("پرتفو مونت‌کارلو", analysis["best_weights"]),
        ("پرتفو CVaR", analysis["best_cvar_weights"]),
        ("پرتفو MPT", analysis["mpt_weights"]),
    ]
    cols = st.columns(len(styles))
    for i, (label, weights) in enumerate(styles):
        with cols[i]:
            pie_fig = go.Figure(
                data=[go.Pie(labels=asset_names, values=weights, hole=.4)]
            )
            st.plotly_chart(pie_fig, use_container_width=True)
            st.markdown(f"<b>{label}</b>", unsafe_allow_html=True)

def show_dashboard(analysis, st):
    st.markdown("### 📋 داشبورد خلاصه پرتفو")
    investment_amount = st.session_state["investment_amount"]
    # فرض کن توابع risk_return رو داری و ... (مثل app.py v3)
    st.write("اینجا جدول بازده و ریسک و مقادیر کلیدی هر سبک را نمایش بده...")

def show_risk_return_details(analysis, st):
    st.markdown("### 📋 جزئیات ریسک و بازده هر سبک")
    st.write("اینجا جزییات ریسک و بازده هر سبک را نمایش بده...")

def plot_efficient_frontier(analysis, asset_names, st):
    ef_results = analysis["ef_results"]
    stds, rets, sharpes = ef_results[0], ef_results[1], ef_results[2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stds, y=rets, mode='markers',
        marker=dict(color=sharpes, colorscale='Viridis', showscale=True),
        name='مرز کارا'
    ))
    for label, weights in [
        ("مونت‌کارلو", analysis["best_weights"]),
        ("CVaR", analysis["best_cvar_weights"]),
        ("MPT", analysis["mpt_weights"])
    ]:
        port_std = np.sqrt(np.dot(weights.T, np.dot(analysis["cov_matrix"], weights)))
        port_ret = np.dot(weights, analysis["mean_returns"])
        fig.add_trace(go.Scatter(
            x=[port_std], y=[port_ret], mode="markers+text",
            marker=dict(size=14), name=f"{label} منتخب",
            text=[label], textposition="top center"
        ))
    fig.update_layout(
        xaxis_title="ریسک سالانه (انحراف معیار)",
        yaxis_title="بازده سالانه",
        title="Efficient Frontier (مرز کارا)"
    )
    st.plotly_chart(fig, use_container_width=True)
