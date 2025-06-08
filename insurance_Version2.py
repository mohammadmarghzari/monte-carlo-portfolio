import numpy as np
import plotly.graph_objects as go

def insurance_effect_on_cov(prices_df, insured_assets):
    cov = prices_df.pct_change().cov()
    for idx, col in enumerate(prices_df.columns):
        info = insured_assets.get(col)
        if info:
            insured_ratio = info['amount'] / max(1e-6, info['base'])
            risk_scale = 1 - info['loss_percent'] / 100 * insured_ratio
            cov.iloc[idx, :] *= risk_scale
            cov.iloc[:, idx] *= risk_scale
    return cov

def show_option_pnl_chart(info):
    x = np.linspace(info['spot'] * 0.5, info['spot'] * 1.5, 400)
    asset_pnl = (x - info['spot']) * info['base']
    put_pnl = np.where(x < info['strike'], (info['strike'] - x) * info['amount'], 0) - info['premium'] * info['amount']
    total_pnl = asset_pnl + put_pnl
    initial_cost = info['spot'] * info['base'] + info['premium'] * info['amount']
    percent_profit = np.where(initial_cost != 0, 100 * total_pnl / initial_cost, 0)
    idx_be = np.argmin(np.abs(total_pnl))
    break_even = x[idx_be]
    break_even_y = total_pnl[idx_be]
    break_even_percent = percent_profit[idx_be]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x[total_pnl>=0], y=total_pnl[total_pnl>=0], mode='lines', name='سود',
        customdata=np.stack([percent_profit[total_pnl>=0]], axis=-1),
        hovertemplate='قیمت: %{x:.3f}<br>سود: %{y:.3f}<br>درصد سود: %{customdata[0]:.2f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=x[total_pnl<0], y=total_pnl[total_pnl<0], mode='lines', name='زیان',
        customdata=np.stack([percent_profit[total_pnl<0]], axis=-1),
        hovertemplate='قیمت: %{x:.3f}<br>زیان: %{y:.3f}<br>درصد زیان: %{customdata[0]:.2f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[break_even], y=[break_even_y], mode='markers+text',
        marker=dict(size=14, color='orange', symbol='x'),
        text=[f'سر به سر\n{break_even:.2f}\n{break_even_percent:.2f}%'],
        textposition="top right", name='نقطه سر به سر',
        hovertemplate='قیمت سر به سر: %{x:.3f}<br>بازده: %{y:.3f}<br>درصد: ' + f'{break_even_percent:.2f}%<extra></extra>'
    ))
    return fig