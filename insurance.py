import streamlit as st
import numpy as np
import plotly.graph_objects as go

def insurance_section(st, asset_names):
    st.subheader("📉 بیمه دارایی‌ها (Married Put)")
    # فرض کن بیمه هر دارایی را از st.session_state["insured_assets"] می‌خوانی
    # و نمودار plotly مثل app.py v3 رسم می‌کنی
    st.write("اینجا فرم پارامترهای بیمه و نمودار سود/زیان بیمه برای هر دارایی را نمایش بده...")
