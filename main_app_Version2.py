import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from portfolio_analysis import analyze_portfolio, get_optimal_weights, get_risk_return_summary, compare_styles_plot
from insurance import insurance_effect_on_cov, show_option_pnl_chart
from ui_helpers import format_money, format_percent, format_float, download_excel, validate_investment_input

st.set_page_config(page_title="تحلیل و شبیه‌سازی پرتفو", layout="wide")
st.title("🧮 ابزار تحلیل و شبیه‌سازی پرتفو")

# نمونه دیتا تستی برای راحتی کاربر
if st.button("🌟 بارگذاری داده تستی"):
    st.session_state["downloaded_dfs"] = [
        ("BTC-USD", pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=12, freq="M"),
                                  "Price": np.linspace(20000, 40000, 12)})),
        ("ETH-USD", pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=12, freq="M"),
                                  "Price": np.linspace(1000, 2500, 12)}))
    ]
    st.session_state["uploaded_dfs"] = []

# بارگذاری و حذف دارایی‌ها (تکمیل کن با کد قبلی خودت!)
# ...

# مقدار سرمایه‌گذاری (ورودی با اعتبارسنجی و فرمت)
st.sidebar.subheader("💵 مقدار کل سرمایه‌گذاری (دلار)")
investment_amount = st.sidebar.text_input(
    "سرمایه کل", value=str(st.session_state.get('investment_amount', 1000)),
    help="عدد را وارد کنید. برای مثال: 2572 یا 0.1"
)
investment_amount_valid, investment_amount_val = validate_investment_input(investment_amount)
if not investment_amount_valid:
    st.sidebar.error("مقدار سرمایه معتبر نیست!")
else:
    st.session_state['investment_amount'] = investment_amount_val
    st.sidebar.markdown(f"مقدار فعلی: <b>{format_money(investment_amount_val)}</b>", unsafe_allow_html=True)

# بیمه (Married Put) و اثر آن روی پرتفو
# ... (مطابق insurance.py و با اعتبارسنجی)

# پارامترهای پرتفو
# ...

# ساخت دیتافریم قیمت‌ها و اسامی دارایی‌ها
if "downloaded_dfs" not in st.session_state: st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state: st.session_state["uploaded_dfs"] = []
if len(st.session_state.get("downloaded_dfs", [])) + len(st.session_state.get("uploaded_dfs", [])) < 2:
    st.warning("حداقل دو دارایی برای تحلیل پرتفو لازم است.")
    st.stop()

asset_names, dfs = [], []
for t, df in st.session_state["downloaded_dfs"]: asset_names.append(t); dfs.append(df)
for t, df in st.session_state["uploaded_dfs"]: asset_names.append(t); dfs.append(df)
# Merge all dataframes on Date
prices_df = dfs[0][['Date', 'Price']].rename(columns={'Price': asset_names[0]}).set_index('Date')
for i in range(1, len(dfs)):
    prices_df = prices_df.join(dfs[i][['Date', 'Price']].rename(columns={'Price': asset_names[i]}).set_index('Date'), how='inner')

# اعمال بیمه روی کوواریانس
insured_assets = st.session_state.get("insured_assets", {})
cov_matrix_adj = insurance_effect_on_cov(prices_df, insured_assets)

# تحلیل پرتفو
portfolio_results = analyze_portfolio(
    prices_df, cov_matrix_adj, st.session_state["investment_amount"], insured_assets
)

# نمایش وزن و مقدار دلاری هر دارایی در سه سبک
st.markdown("### 💰 ترکیب سرمایه‌گذاری هر سبک")
for style in ["Monte Carlo", "CVaR", "MPT"]:
    st.markdown(f"**{style}:**")
    weights = get_optimal_weights(portfolio_results, style)
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

# نمایش خلاصه ریسک و بازده
st.markdown("### 📊 داشبورد خلاصه پرتفو")
risk_return_summary = get_risk_return_summary(portfolio_results)
st.dataframe(risk_return_summary, use_container_width=True)

# نمایش نمودار مقایسه‌ای ریسک-بازده سه سبک
st.markdown("### 📈 مقایسه بصری سبک‌های پرتفو")
fig_compare = compare_styles_plot(portfolio_results, asset_names)
st.plotly_chart(fig_compare, use_container_width=True)

# نمایش نمودار سود/زیان بیمه برای هر دارایی (با درصد و نقطه سر به سر دقیق)
st.markdown("### 📉 بیمه دارایی‌ها (Married Put)")
for name in insured_assets:
    info = insured_assets[name]
    fig_option = show_option_pnl_chart(info)
    st.markdown(f"<b>{name}</b>", unsafe_allow_html=True)
    st.plotly_chart(fig_option, use_container_width=True)

# خروجی گرفتن: Excel
st.markdown("### 📤 ذخیره گزارش")
excel_buffer = download_excel(portfolio_results, asset_names)
st.download_button("دانلود خروجی Excel", excel_buffer, file_name="portfolio_report.xlsx")
st.info("برای خروجی PDF از دکمه print مرورگر یا افزونه PDF استفاده کنید (یا درخواست دهید تا نمونه کد ارائه شود).")

# راهنمای کاربری و FAQ
with st.expander("📚 راهنمای استفاده و سوالات متداول"):
    st.markdown("""
    - برای شروع، داده‌های دارایی‌های خود را بارگذاری یا دانلود کنید.
    - بیمه اختیاری است اما اثر آن به صورت دقیق روی ریسک پرتفو لحاظ می‌شود.
    - می‌توانید وزن هر دارایی را به‌صورت دستی وارد کنید (در آینده).
    - خروجی Excel شامل تمام وزن‌ها و بازده‌هاست.
    """)