import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def run_portfolio_analysis(prices_df, asset_names, investment_amount):
    # پارامترهای ساده‌شده
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

    # سبک‌های مختلف: مونت‌کارلو، CVaR، MPT
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

def plot_pie_charts(analysis, asset_names, investment_amount):
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
            for asset, w in zip(asset_names, weights):
                st.markdown(f"{asset}: {w*100:.2f}% ({investment_amount*w:.2f} $)")

def plot_efficient_frontiers(analysis, asset_names):
    ef_results = analysis["ef_results"]
    stds, rets, sharpes = ef_results[0], ef_results[1], ef_results[2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stds, y=rets, mode='markers',
        marker=dict(color=sharpes, colorscale='Viridis', showscale=True),
        name='مرز کارا'
    ))
    # مارکر پرتفوهای منتخب
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

def plot_mpt_efficient_frontier(analysis, asset_names):
    ef_results = analysis["ef_results"]
    stds, rets, sharpes = ef_results[0], ef_results[1], ef_results[2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stds, y=rets, mode='lines+markers',
        line=dict(color='blue', width=2), name='مرز کارا MPT'
    ))
    idx_max_sharpe = np.argmax(sharpes)
    fig.add_trace(go.Scatter(
        x=[stds[idx_max_sharpe]], y=[rets[idx_max_sharpe]],
        mode="markers+text", marker=dict(size=16, color='red'),
        text=["بیشینه شارپ"], textposition="top center", name="بیشینه شارپ"
    ))
    fig.update_layout(
        xaxis_title="ریسک سالانه", yaxis_title="بازده سالانه",
        title="Efficient Frontier ویژه MPT"
    )
    st.plotly_chart(fig, use_container_width=True)

def price_forecast_section(prices_df, asset_names):
    import warnings
    warnings.filterwarnings('ignore')
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        st.warning("پیش‌بینی قیمت نیاز به statsmodels دارد. با دستور pip install statsmodels نصب کنید.")
        return
    steps = st.slider("تعداد ماه پیش‌بینی", 3, 36, 12)
    cols = st.columns(len(asset_names))
    for i, name in enumerate(asset_names):
        with cols[i]:
            ts = prices_df[name].resample("M").last().dropna()
            if len(ts) < 15:
                st.warning(f"تاریخچه {name} کوتاه است!")
                continue
            model = ARIMA(ts, order=(1,1,1))
            fit = model.fit()
            pred = fit.get_forecast(steps=steps)
            forecast = pred.predicted_mean
            ci = pred.conf_int()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts, mode="lines", name="تاریخی"))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode="lines", name="پیش‌بینی"))
            fig.add_trace(go.Scatter(
                x=forecast.index.tolist()+forecast.index[::-1].tolist(),
                y=ci.iloc[:,0].tolist()+ci.iloc[:,1].tolist()[::-1],
                fill="toself", fillcolor="rgba(0,100,200,0.2)", line=dict(color="rgba(255,255,255,0)"),
                showlegend=False, name="بازه اطمینان"
            ))
            fig.update_layout(title=f"پیش‌بینی قیمت آینده {name}", xaxis_title="تاریخ", yaxis_title="قیمت")
            st.plotly_chart(fig, use_container_width=True)
