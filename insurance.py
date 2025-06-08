import streamlit as st
import numpy as np
import plotly.graph_objects as go

def insurance_input_sidebar(name):
    with st.sidebar.expander(f"⚙️ بیمه برای {name}", expanded=False):
        st.markdown("""
        <div dir="rtl" style="text-align: right;">
        <b>Married Put چیست؟</b>
        <br>بیمه در پرتفو (Married Put) یعنی شما همزمان با نگهداری دارایی، یک قرارداد اختیار فروش (Put Option) برای همان دارایی [...]
        </div>
        """, unsafe_allow_html=True)
        insured = st.checkbox(f"فعال‌سازی بیمه برای {name}", key=f"insured_{name}")
        if insured:
            loss_percent = st.number_input(f"📉 درصد ضرر معامله پوت برای {name}", 0.0, 100.0, 30.0, step=0.01, format="%.3f", key=f"loss_{name}")
            strike = st.number_input(f"🎯 قیمت اعمال پوت برای {name}", 0.0, 1e6, 100.0, step=0.01, format="%.3f", key=f"strike_{name}")
            premium = st.number_input(f"💰 قیمت قرارداد پوت برای {name}", 0.0, 1e6, 5.0, step=0.01, format="%.3f", key=f"premium_{name}")
            amount = st.number_input(f"📦 مقدار قرارداد برای {name}", 0.0, 1e6, 1.0, step=0.01, format="%.3f", key=f"amount_{name}")
            spot_price = st.number_input(f"📌 قیمت فعلی دارایی پایه {name}", 0.0, 1e6, 100.0, step=0.01, format="%.3f", key=f"spot_{name}")
            asset_amount = st.number_input(f"📦 مقدار دارایی پایه {name}", 0.0, 1e6, 1.0, step=0.01, format="%.3f", key=f"base_{name}")
            st.session_state["insured_assets"][name] = {
                'loss_percent': loss_percent,
                'strike': strike,
                'premium': premium,
                'amount': amount,
                'spot': spot_price,
                'base': asset_amount
            }
        else:
            st.session_state["insured_assets"].pop(name, None)

def plot_married_put(name):
    info = st.session_state["insured_assets"][name]
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
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x[total_pnl>=0],
        y=total_pnl[total_pnl>=0],
        mode='lines',
        name='سود',
        line=dict(color='green', width=3),
        customdata=np.stack([percent_profit[total_pnl>=0]], axis=-1),
        hovertemplate='قیمت: %{x:.3f}<br>سود: %{y:.3f}<br>درصد سود: %{customdata[0]:.2f}%<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=x[total_pnl<0],
        y=total_pnl[total_pnl<0],
        mode='lines',
        name='زیان',
        line=dict(color='red', width=3),
        customdata=np.stack([percent_profit[total_pnl<0]], axis=-1),
        hovertemplate='قیمت: %{x:.3f}<br>زیان: %{y:.3f}<br>درصد زیان: %{customdata[0]:.2f}%<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=x, y=asset_pnl, mode='lines', name='دارایی پایه', line=dict(dash='dot', color='gray')
    ))
    fig2.add_trace(go.Scatter(
        x=x, y=put_pnl, mode='lines', name='پوت', line=dict(dash='dot', color='blue')
    ))
    fig2.add_trace(go.Scatter(
        x=[break_even], y=[break_even_y], mode='markers+text',
        marker=dict(size=14, color='orange', symbol='x'),
        text=[f'سر به سر\n{break_even:.2f}\n{break_even_percent:.2f}%'],
        textposition="top right",
        name='نقطه سر به سر',
        hovertemplate='قیمت سر به سر: %{x:.3f}<br>بازده: %{y:.3f}<br>درصد: ' + f'{break_even_percent:.2f}%<extra></extra>'
    ))
    st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
