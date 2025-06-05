import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import yfinance as yf
import json
import base64

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو، CVaR و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو، CVaR و استراتژی Married Put")

# Function to calculate the optimal portfolio based on different optimization methods
def optimize_portfolio(method, returns, cov_matrix, mean_returns, n_portfolios, cvar_alpha=0.95):
    results = np.zeros((5 + len(returns.columns), n_portfolios))
    np.random.seed(42)
    
    for i in range(n_portfolios):
        valid_weights = False
        while not valid_weights:
            weights = np.random.random(len(returns.columns))
            weights /= np.sum(weights)
            port_return = np.dot(weights, mean_returns)
            port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if method == 'max_sharpe':
                # Maximize Sharpe Ratio
                sharpe_ratio = port_return / port_std
                results[0, i] = port_return
                results[1, i] = port_std
                results[2, i] = sharpe_ratio
            elif method == 'max_sortino':
                # Maximize Sortino Ratio
                downside_risk = np.sqrt(np.dot(weights.T, np.dot(np.minimum(returns, 0).cov(), weights)))
                sortino_ratio = port_return / downside_risk if downside_risk > 0 else np.nan
                results[0, i] = port_return
                results[1, i] = port_std
                results[3, i] = sortino_ratio
            elif method == 'min_cvar':
                # Minimize CVaR
                mc_sims = np.random.multivariate_normal(mean_returns, cov_matrix, 1000)
                port_mc_returns = np.dot(mc_sims, weights)
                VaR = np.percentile(port_mc_returns, (1 - cvar_alpha) * 100)
                CVaR = port_mc_returns[port_mc_returns <= VaR].mean() if np.any(port_mc_returns <= VaR) else VaR
                results[0, i] = port_return
                results[1, i] = port_std
                results[4, i] = -CVaR
            # Add more optimization criteria as needed.
    return results

# Existing data handling and file reading code is assumed to be here

# Assuming 'returns', 'cov_matrix', and 'mean_returns' have been calculated
# Call the optimization functions with method arguments like 'max_sharpe', 'max_sortino', 'min_cvar'
results_max_sharpe = optimize_portfolio('max_sharpe', returns, cov_matrix, mean_returns, n_portfolios=10000)
results_max_sortino = optimize_portfolio('max_sortino', returns, cov_matrix, mean_returns, n_portfolios=10000)
results_min_cvar = optimize_portfolio('min_cvar', returns, cov_matrix, mean_returns, n_portfolios=10000)

# Now plotting the results
st.subheader("📈 بهترین پرتفو بر اساس نسبت شارپ")
best_sharpe_idx = np.argmax(results_max_sharpe[2])
st.markdown(f"🔹 بازده: {results_max_sharpe[0, best_sharpe_idx]:.2%}")
st.markdown(f"⚠️ ریسک: {results_max_sharpe[1, best_sharpe_idx]:.2%}")
st.markdown(f"🧠 نسبت شارپ: {results_max_sharpe[2, best_sharpe_idx]:.2f}")

# Add similar sections for Sortino and CVaR

# Saving the file for download
portfolio_state = {
    "best_sharpe_weights": results_max_sharpe[5:, best_sharpe_idx].tolist(),
    "best_sortino_weights": results_max_sortino[5:, best_sortino_idx].tolist(),
    "best_cvar_weights": results_min_cvar[5:, best_cvar_idx].tolist(),
    "best_sharpe_return": float(results_max_sharpe[0, best_sharpe_idx]),
    "best_sharpe_risk": float(results_max_sharpe[1, best_sharpe_idx]),
    "best_sharpe_sharpe": float(results_max_sharpe[2, best_sharpe_idx]),
    "best_sortino_return": float(results_max_sortino[0, best_sortino_idx]),
    "best_sortino_risk": float(results_max_sortino[1, best_sortino_idx]),
    "best_sortino_sortino": float(results_max_sortino[3, best_sortino_idx]),
    "best_cvar_return": float(results_min_cvar[0, best_cvar_idx]),
    "best_cvar_risk": float(results_min_cvar[1, best_cvar_idx]),
    "best_cvar_cvar": float(results_min_cvar[4, best_cvar_idx]),
}

portfolio_json = json.dumps(portfolio_state, ensure_ascii=False, indent=2)
b64 = base64.b64encode(portfolio_json.encode()).decode()
st.sidebar.markdown(
    f'<a download="portfolio_state.json" href="data:application/json;base64,{b64}" target="_blank">⬇️ ذخیره فایل پرتفو (JSON)</a>',
    unsafe_allow_html=True
)

# Continue with other visualization and outputs as per the existing structure
