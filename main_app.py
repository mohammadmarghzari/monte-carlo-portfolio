import streamlit as st
from data_handlers import get_dataframes
from portfolio_analysis import (
    run_portfolio_analysis,
    show_weights,
    show_dashboard,
    show_risk_return_details,
    plot_pie_charts,
    plot_efficient_frontier,
)
from insurance import insurance_section

st.set_page_config(page_title="My Portfolio App", layout="wide")
st.title("My Portfolio App")

# بارگزاری و آماده‌سازی داده‌ها
prices_df, asset_names = get_dataframes(st)

if prices_df is not None and len(asset_names) > 1:
    # تحلیل و محاسبه پرتفوها
    analysis = run_portfolio_analysis(prices_df, asset_names, st)

    # نمایش جداول و نمودارها
    show_weights(analysis, asset_names, st)
    plot_pie_charts(analysis, asset_names, st)
    show_dashboard(analysis, st)
    show_risk_return_details(analysis, st)
    plot_efficient_frontier(analysis, asset_names, st)

    # بخش بیمه
    insurance_section(st, asset_names)
else:
    st.warning("لطفاً حداقل داده معتبر برای دو دارایی وارد کنید یا دانلود کنید.")
