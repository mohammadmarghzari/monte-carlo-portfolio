import numpy as np
import pandas as pd

def analyze_portfolio(prices_df, cov_matrix_adj, investment_amount, insured_assets):
    # محاسبات پرتفو (مارکویتز، مونت‌کارلو، CVaR)
    # برای سادگی نسخه پایه (می‌توان حرفه‌ای‌ترش کرد)
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean() * 12
    cov_matrix = cov_matrix_adj * 12
    n_assets = len(prices_df.columns)
    n_portfolios = 2000
    results = {"weights": {}, "risk": {}, "return": {}, "risk_m": {}, "return_m": {}}
    all_weights, all_returns, all_risks = [], [], []
    np.random.seed(42)
    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        all_weights.append(weights)
        all_returns.append(port_return)
        all_risks.append(port_risk)
    all_weights = np.array(all_weights)
    all_returns = np.array(all_returns)
    all_risks = np.array(all_risks)
    # Monte Carlo: نزدیک‌ترین پرتفوی به ریسک متوسط
    idx_mc = np.argmin(np.abs(all_risks - all_risks.mean()))
    # MPT: بیشترین نسبت شارپ
    sharpe = all_returns / all_risks
    idx_mpt = np.argmax(sharpe)
    # CVaR: کمترین میانگین زیان پایین
    idx_cvar = np.argmin(all_returns - 2 * all_risks)
    # ماهانه
    mean_month = mean_returns / 12
    risk_month = np.sqrt(np.diag(cov_matrix)) / np.sqrt(12)
    for style, idx in zip(["Monte Carlo", "MPT", "CVaR"], [idx_mc, idx_mpt, idx_cvar]):
        w = all_weights[idx]
        pf_monthly = (prices_df * w).sum(axis=1).pct_change().dropna()
        results["weights"][style] = w
        results["return"][style] = all_returns[idx]
        results["risk"][style] = all_risks[idx]
        results["return_m"][style] = pf_monthly.mean()
        results["risk_m"][style] = pf_monthly.std()
    return results

def get_optimal_weights(results, style):
    return results["weights"][style]

def get_risk_return_summary(results):
    rows = []
    for style in ["Monte Carlo", "CVaR", "MPT"]:
        rows.append({
            "سبک": style,
            "بازده سالانه": results["return"][style],
            "ریسک سالانه": results["risk"][style],
            "بازده ماهانه": results["return_m"][style],
            "ریسک ماهانه": results["risk_m"][style],
        })
    return pd.DataFrame(rows)

def compare_styles_plot(results, asset_names):
    import plotly.graph_objects as go
    styles = ["Monte Carlo", "CVaR", "MPT"]
    colors = ["blue", "red", "green"]
    fig = go.Figure()
    for style, color in zip(styles, colors):
        fig.add_trace(go.Scatter(
            x=[results["risk"][style]*100],
            y=[results["return"][style]*100],
            mode='markers+text',
            marker=dict(size=18, color=color),
            text=[style],
            textposition="top center"
        ))
    fig.update_layout(title="مقایسه ریسک و بازده سبک‌ها", xaxis_title="ریسک (%)", yaxis_title="بازده (%)")
    return fig
