import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="تحلیل پرتفو با بیمه و پیش‌بینی", layout="wide")
st.markdown("<h1 style='text-align: right;'>📊 ابزار جامع تحلیل پرتفو، بیمه و پیش‌بینی قیمت</h1>", unsafe_allow_html=True)

# === ۱. بازه زمانی تحلیل ===
st.sidebar.markdown("<h3 style='text-align: right;'>۱. بازه زمانی تحلیل</h3>", unsafe_allow_html=True)
date_range_mode = st.sidebar.radio("انتخاب بازه:", ["کل داده", "بازه دلخواه"], horizontal=True)
date_start, date_end = None, None
if date_range_mode == "بازه دلخواه":
    date_start = st.sidebar.date_input("تاریخ شروع", value=datetime(2022,1,1))
    date_end = st.sidebar.date_input("تاریخ پایان", value=datetime.now())
    if date_end < date_start:
        st.sidebar.error("تاریخ پایان باید بعد از تاریخ شروع باشد.")

# === ۲. بارگذاری و تنظیم دارایی‌ها و بیمه ===
uploaded_files = st.sidebar.file_uploader("آپلود فایل‌های CSV با ستون‌های Date و Price", type="csv", accept_multiple_files=True)
asset_settings = {}
insurance_settings = {}

if uploaded_files:
    st.sidebar.markdown("<h3 style='text-align: right;'>۲. تنظیمات وزن، محدودیت و بیمه هر دارایی</h3>", unsafe_allow_html=True)
    for file in uploaded_files:
        asset_name = file.name.split('.')[0]
        st.sidebar.markdown(f"<b style='text-align: right;'>{asset_name}</b>", unsafe_allow_html=True)
        min_w = st.sidebar.slider(f"حداقل وزن {asset_name} (%)", 0.0, 100.0, 0.0, 1.0, key=f"min_{asset_name}")/100
        max_w = st.sidebar.slider(f"حداکثر وزن {asset_name} (%)", 0.0, 100.0, 100.0, 1.0, key=f"max_{asset_name}")/100
        init_w = st.sidebar.slider(f"وزن اولیه {asset_name} (%)", min_w*100, max_w*100, ((min_w+max_w)/2)*100, 1.0, key=f"init_{asset_name}")/100
        insure = st.sidebar.checkbox(f"فعالسازی بیمه برای {asset_name}", key=f"insure_{asset_name}")
        if insure:
            strat_type = st.sidebar.selectbox(f"نوع بیمه {asset_name}", ["مرید پوت (Protective Put)", "پریوت پوت (Perpetual Put)"], key=f"stype_{asset_name}")
            entry_price = st.sidebar.number_input(
                f"قیمت خرید دارایی پایه ({asset_name})",
                min_value=0.00001, max_value=1e7,
                value=1000.0, step=0.00001, format="%.5f",
                key=f"entry_{asset_name}"
            )
            strike = st.sidebar.number_input(
                f"قیمت اعمال (استرایک) ({asset_name})",
                min_value=0.00001, max_value=1e7,
                value=900.0, step=0.00001, format="%.5f",
                key=f"strike_{asset_name}"
            )
            premium = st.sidebar.number_input(
                f"پریمیوم (حق بیمه) ({asset_name})",
                min_value=0.00001, max_value=1e7,
                value=1.0, step=0.00001, format="%.5f",
                key=f"premium_{asset_name}"
            )
            pos_size = st.sidebar.number_input(
                f"مقدار دارایی پایه ({asset_name})",
                min_value=0.00001, max_value=1e7,
                value=1.0, step=0.00001, format="%.5f",
                key=f"possize_{asset_name}"
            )
            opt_size = st.sidebar.number_input(
                f"مقدار قرارداد آپشن ({asset_name})",
                min_value=0.00001, max_value=1e7,
                value=1.0, step=0.00001, format="%.5f",
                key=f"optsize_{asset_name}"
            )
            insurance_settings[asset_name] = {
                "active": True,
                "strat": strat_type,
                "entry": entry_price,
                "strike": strike,
                "premium": premium,
                "pos_size": pos_size,
                "opt_size": opt_size
            }
        else:
            insurance_settings[asset_name] = {"active": False}
        asset_settings[asset_name] = {"min": min_w, "max": max_w, "init": init_w}

# === ۳. تنظیمات بازه بازده ===
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: right;'>۳. تنظیمات بازه بازده</h3>", unsafe_allow_html=True)
period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]

# === ۴. انتخاب روش بهینه‌سازی ===
st.sidebar.markdown("---")
st.sidebar.header("🎯 روش بهینه‌سازی پرتفو")
all_methods = [
    "واریانس میانگین (مرز کارا)",
    "کمینه CVaR (ارزش در معرض ریسک مشروط)",
    "برابری ریسک",
    "حداقل خطای ردیابی",
    "بیشینه نسبت اطلاعات",
    "بیشینه نرخ رشد هندسی (Kelly)",
    "بیشینه نسبت سورتینو",
    "بیشینه نسبت امگا",
    "کمترین افت سرمایه",
    "Black-Litterman"
]
method = st.sidebar.selectbox("روش بهینه‌سازی:", all_methods)
st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ پارامترهای تخصصی هر روش"):
    if method in ["کمینه CVaR (ارزش در معرض ریسک مشروط)"]:
        cvar_alpha = st.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)
    else:
        cvar_alpha = 0.95
    if method in ["بیشینه نسبت سورتینو", "بیشینه نسبت امگا", "کمترین افت سرمایه"]:
        target_return = st.slider("حداقل بازده قابل قبول (%)", 0.0, 20.0, 5.0, 0.5)/100
    else:
        target_return = 0.0
    n_portfolios = st.slider("تعداد شبیه‌سازی پرتفوها", 1000, 10000, 3000, 1000)

# === توابع بیمه ===
def protective_put_payoff(price, entry, strike, premium, pos_size, opt_size):
    pl_stock = (price - entry) * pos_size
    pl_option = np.maximum(strike - price, 0) * opt_size
    cost = premium * opt_size
    total_pl = pl_stock + pl_option - cost
    pct = total_pl / (entry * pos_size)
    return pct

def perpetual_put_payoff(price, entry, strike, premium, pos_size, opt_size):
    pl_stock = (price - entry) * pos_size
    pl_option = np.maximum(strike - price, 0) * opt_size
    n_periods = np.maximum(np.floor((entry - price) / (entry - strike)), 1)
    cost = premium * opt_size * n_periods
    total_pl = pl_stock + pl_option - cost
    pct = total_pl / (entry * pos_size)
    return pct

def insured_returns(prices, asset_name):
    info = insurance_settings.get(asset_name, {})
    if not info.get("active", False):
        return prices.pct_change().dropna()
    strat = info["strat"]
    entry = info["entry"]
    strike = info["strike"]
    premium = info["premium"]
    pos_size = info["pos_size"]
    opt_size = info["opt_size"]
    if strat == "مرید پوت (Protective Put)":
        payoff = protective_put_payoff(prices, entry, strike, premium, pos_size, opt_size)
    else:
        payoff = perpetual_put_payoff(prices, entry, strike, premium, pos_size, opt_size)
    returns = np.diff(payoff)
    return pd.Series(returns, index=prices.index[1:])

# === ۵. پردازش داده و محاسبات ===
prices_df = pd.DataFrame()
asset_names = []
if uploaded_files:
    for file in uploaded_files:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower()
        if 'date' not in df.columns or 'price' not in df.columns:
            st.error(f"فایل {file.name} ستون‌های 'Date' و 'Price' ندارد.")
            continue
        df['Date'] = pd.to_datetime(df['date'])
        df['Price'] = pd.to_numeric(df['price'].astype(str).str.replace(",", ""), errors='coerce')
        df = df.dropna().set_index('Date')
        if date_range_mode == "بازه دلخواه":
            df = df.loc[(df.index >= pd.to_datetime(date_start)) & (df.index <= pd.to_datetime(date_end))]
        df = df[['Price']]
        name = file.name.split('.')[0]
        df.columns = [name]
        asset_names.append(name)
        if prices_df.empty:
            prices_df = df
        else:
            prices_df = prices_df.join(df, how='inner')
    prices_df = prices_df.dropna()

if not prices_df.empty:
    st.markdown("<h2 style='text-align: right;'>📈 قیمت‌های تعدیل‌شده دارایی‌ها</h2>", unsafe_allow_html=True)
    st.dataframe(prices_df.tail())

    insured_ret_df = pd.DataFrame(index=prices_df.index[1:])
    for name in asset_names:
        insured_ret_df[name] = insured_returns(prices_df[name], name)
    insured_ret_df = insured_ret_df.dropna()
    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    insured_ret_resampled = insured_ret_df.resample(resample_rule).last().dropna()
    mean_returns = insured_ret_resampled.mean() * annual_factor
    cov_matrix = insured_ret_resampled.cov() * annual_factor

    st.markdown("<h2 style='text-align: right;'>📊 بازده دارایی‌ها (با لحاظ بیمه در صورت فعال‌سازی)</h2>", unsafe_allow_html=True)
    st.dataframe(insured_ret_resampled.tail())

    tracking_index = insured_ret_resampled.mean(axis=1).values

    results = []
    weights_all = []
    for w in np.random.dirichlet(np.ones(len(asset_names)), n_portfolios):
        legal = True
        for i, name in enumerate(asset_names):
            min_w, max_w = asset_settings[name]["min"], asset_settings[name]["max"]
            if not (min_w <= w[i] <= max_w):
                legal = False
        if not legal:
            continue
        port_ret = np.dot(w, mean_returns)
        port_risk = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sims = np.random.multivariate_normal(mean_returns / annual_factor, cov_matrix / annual_factor, 1000)
        sim_returns = np.dot(sims, w)
        var = np.percentile(sim_returns, (1 - cvar_alpha) * 100)
        cvar = sim_returns[sim_returns <= var].mean() if np.any(sim_returns <= var) else var
        downside = insured_ret_resampled.copy()
        downside[downside > target_return] = 0
        downside_std = np.sqrt(np.dot(w.T, np.dot(downside.cov() * annual_factor, w)))
        sortino = (port_ret - target_return) / downside_std if downside_std > 0 else 0
        port_returns = np.dot(insured_ret_resampled.values, w)
        omega = np.sum(np.maximum(port_returns - target_return, 0)) / (np.abs(np.sum(np.minimum(port_returns - target_return, 0))) + 1e-8)
        cum = (1 + port_returns).cumprod()
        peak = np.maximum.accumulate(cum)
        drawdowns = (cum - peak) / peak
        max_drawdown = drawdowns.min()
        excess = insured_ret_resampled.sub(mean_returns.mean(), axis=1)
        info = (port_ret - mean_returns.mean()) / (np.std(np.dot(excess.values, w)) + 1e-8)
        kelly_growth = np.mean(np.log1p(port_returns))
        contrib = w * np.dot(cov_matrix, w)
        risk_budget = np.std(contrib) / (np.mean(contrib) + 1e-8)
        tracking_err = np.std(port_returns - tracking_index)
        results.append({
            "weights": w, "return": port_ret, "risk": port_risk, "cvar": cvar, "sortino": sortino,
            "omega": omega, "drawdown": max_drawdown, "info": info, "kelly": kelly_growth,
            "risk_budget": risk_budget, "tracking": tracking_err
        })
        weights_all.append(w)

    # ===== انتخاب پرتفو بهینه =====
    if method == "واریانس میانگین (مرز کارا)":
        best_index = np.argmax([r["return"]/r["risk"] for r in results])
        explain = "در این سبک، هدف ما بیشترین نسبت بازده به ریسک (Sharpe Ratio) است. مرز کارا نشان‌دهنده بهترین ترکیب بازده و ریسک برای هر سطح ریسک است."
    elif method == "کمینه CVaR (ارزش در معرض ریسک مشروط)":
        best_index = np.argmin([r["cvar"] for r in results])
        explain = "این روش پرتفو با کمترین ارزش در معرض ریسک مشروط (CVaR) را انتخاب می‌کند تا احتمال ضرر شدید کاهش یابد."
    elif method == "برابری ریسک":
        best_index = np.argmin([r["risk_budget"] for r in results])
        explain = "در این سبک، هدف مساوی‌کردن سهم ریسک دارایی‌ها در پرتفو است."
    elif method == "حداقل خطای ردیابی":
        best_index = np.argmin([r["tracking"] for r in results])
        explain = "هدف کاهش انحراف پرتفو نسبت به شاخص معیار (مثلاً میانگین بازار) است."
    elif method == "بیشینه نسبت اطلاعات":
        best_index = np.argmax([r["info"] for r in results])
        explain = "پرتفویی انتخاب می‌شود که بیشترین نسبت اطلاعات (Information Ratio) را نسبت به پرتفو معیار داشته باشد."
    elif method == "بیشینه نرخ رشد هندسی (Kelly)":
        best_index = np.argmax([r["kelly"] for r in results])
        explain = "در این روش، بیشترین نرخ رشد هندسی سرمایه (Kelly Criterion) هدف است."
    elif method == "بیشینه نسبت سورتینو":
        best_index = np.argmax([r["sortino"] for r in results])
        explain = "این سبک پرتفو با بیشترین نسبت سورتینو (توجه ویژه به زیان‌های پایین‌تر از بازده هدف) را انتخاب می‌کند."
    elif method == "بیشینه نسبت امگا":
        best_index = np.argmax([r["omega"] for r in results])
        explain = "این روش پرتفو با بیشترین نسبت امگا (Omega) را انتخاب می‌کند که نشان‌دهنده پتانسیل سود به زیان است."
    elif method == "کمترین افت سرمایه":
        best_index = np.argmax([r["drawdown"] for r in results])
        explain = "در این سبک، پرتفو با کمترین افت سرمایه (Drawdown) انتخاب می‌شود تا کاهش سرمایه به حداقل برسد."
    elif method == "Black-Litterman":
        best_index = np.argmax([r["return"]/r["risk"] for r in results])
        explain = "در این مدل دیدگاه‌های کاربر لحاظ می‌شود (در این نسخه: مشابه مرز کارا)."
    else:
        best_index = 0
        explain = "-"

    best = results[best_index]
    best_weights = weights_all[best_index]

    st.markdown(f"<div style='text-align:right;'><b>روش بهینه‌سازی:</b> {method}</div>", unsafe_allow_html=True)
    st.markdown(f"<div dir='rtl'>{explain}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>📈 <b>بازده پرتفو:</b> {best['return']:.2%}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>⚠️ <b>ریسک پرتفو:</b> {best['risk']:.2%}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>📉 <b>CVaR:</b> {best['cvar']:.2%}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>📊 <b>نسبت سورتینو:</b> {best['sortino']:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>ℹ️ <b>نسبت اطلاعات:</b> {best['info']:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>📈 <b>رشد هندسی (Kelly):</b> {best['kelly']:.2%}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>📉 <b>حداکثر افت سرمایه:</b> {best['drawdown']:.2%}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>⚖️ <b>نسبت امگا:</b> {best['omega']:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>📊 <b>تنوع ریسک:</b> {best['risk_budget']:.2f}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;'>🔁 <b>خطای ردیابی:</b> {best['tracking']:.4f}</div>", unsafe_allow_html=True)

    # ==== نمایش وزن پرتفو بهینه ====
    st.markdown("<h3 style='text-align:right;'>📌 نمودار توزیع وزن پرتفو بهینه</h3>", unsafe_allow_html=True)
    fig = go.Figure(data=[go.Pie(labels=asset_names, values=best['weights'], hole=0.5)])
    fig.update_layout(title="توزیع وزن دارایی‌ها")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "<div style='text-align:right;'>این نمودار سهم هر دارایی در پرتفو بهینه را نمایش می‌دهد. درصد وزنی هر دارایی در دایره مشخص است.</div>",
        unsafe_allow_html=True
    )

    # ==== مرز کارا و موقعیت پرتفو بهینه ====
    st.markdown("<h3 style='text-align:right;'>🌈 نمودار مرز کارا با موقعیت پرتفو بهینه</h3>", unsafe_allow_html=True)
    df = pd.DataFrame(results)
    fig2 = px.scatter(df, x="risk", y="return", color="sortino",
                      hover_data=["info", "omega", "cvar"], title="مرز کارا با رنگ‌بندی نسبت سورتینو")
    fig2.add_trace(go.Scatter(
        x=[best["risk"]], y=[best["return"]],
        mode="markers+text",
        marker=dict(size=16, color="red"),
        text=["پرتفو بهینه"],
        textposition="top center",
        name="پرتفو بهینه"
    ))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "<div style='text-align:right;'>هر نقطه یک پرتفو شبیه‌سازی‌شده را نشان می‌دهد. نقطه قرمز موقعیت پرتفو بهینه بر اساس سبک انتخابی است.</div>",
        unsafe_allow_html=True
    )

    # ==== جدول وزن پرتفو ====
    st.markdown("<h3 style='text-align:right;'>📋 جدول وزن پرتفو بهینه</h3>", unsafe_allow_html=True)
    table = pd.DataFrame({"دارایی": asset_names, "وزن بهینه (%)": np.array(best['weights'])*100,
                          "حداقل وزن (%)": [asset_settings[n]["min"]*100 for n in asset_names],
                          "حداکثر وزن (%)": [asset_settings[n]["max"]*100 for n in asset_names]})
    st.dataframe(table.set_index("دارایی"), use_container_width=True, height=300)
    st.markdown("<div style='text-align:right;'>وزن بهینه هر دارایی به درصد و محدودیت‌های تعیین‌شده در جدول بالا نمایش داده شده است.</div>", unsafe_allow_html=True)

    # ==== نمودار استراتژی بیمه برای هر دارایی ====
    st.markdown("<h3 style='text-align:right;'>🛡️ نمودار سود/زیان استراتژی بیمه (مرید پوت / پریوت پوت)</h3>", unsafe_allow_html=True)
    for name in asset_names:
        info = insurance_settings.get(name, {})
        if info.get("active", False):
            strat = info["strat"]
            entry = info["entry"]
            strike = info["strike"]
            premium = info["premium"]
            pos_size = info["pos_size"]
            opt_size = info["opt_size"]
            price_range = np.linspace(0.7*entry, 1.3*entry, 200)
            if strat == "مرید پوت (Protective Put)":
                profit = protective_put_payoff(price_range, entry, strike, premium, pos_size, opt_size)
            else:
                profit = perpetual_put_payoff(price_range, entry, strike, premium, pos_size, opt_size)
            idx_cross = np.argmin(np.abs(profit))
            breakeven = price_range[idx_cross]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=price_range, y=profit*100, mode="lines",
                line=dict(color="green"),
                name="سود/زیان استراتژی"
            ))
            green_x = price_range[profit >= 0]
            green_y = profit[profit >= 0]*100
            if len(green_x) > 0:
                fig3.add_trace(go.Scatter(
                    x=green_x, y=green_y, mode='lines',
                    fill='tozeroy', fillcolor='rgba(0,255,0,0.15)', line=dict(color="rgba(0,0,0,0)"),
                    name="ناحیه سود"
                ))
            red_x = price_range[profit < 0]
            red_y = profit[profit < 0]*100
            if len(red_x) > 0:
                fig3.add_trace(go.Scatter(
                    x=red_x, y=red_y, mode='lines',
                    fill='tozeroy', fillcolor='rgba(255,0,0,0.15)', line=dict(color="rgba(0,0,0,0)"),
                    name="ناحیه زیان"
                ))
            fig3.add_shape(type="line", x0=breakeven, x1=breakeven, y0=min(profit*100), y1=max(profit*100),
                           line=dict(color="blue", width=2, dash="dash"), name="سربه‌سر")
            fig3.add_annotation(
                x=breakeven, y=0, text="نقطه سربه‌سر", showarrow=True, arrowhead=1, yshift=10, font=dict(color="blue")
            )
            fig3.update_layout(
                xaxis_title="قیمت دارایی پایه",
                yaxis_title="درصد سود/زیان (%)",
                title=f"استراتژی بیمه برای {name}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(tickformat=".4f"),
                yaxis=dict(tickformat=".2f")
            )
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(
                f"""<div style='text-align:right;'>
                <b>توضیح:</b> محور افقی قیمت دارایی، محور عمودی درصد سود/زیان کل است. بالاتر از خط سربه‌سر سبز و پایین‌تر قرمز. بیمه مرید پوت زیان را محدود و سود را بعد از کسر حق بیمه نشان می‌دهد.
                </div>""", unsafe_allow_html=True)

    # ==== پیش‌بینی قیمت برای پرتفو بهینه ====
    st.markdown("<h3 style='text-align:right;'>🔮 پیش‌بینی کمترین و بیشترین قیمت پرتفو بهینه</h3>", unsafe_allow_html=True)
    # با فرض توزیع نرمال بازده پرتفو بهینه
    mu = best['return'] / annual_factor
    sigma = best['risk'] / np.sqrt(annual_factor)
    last_prices = resampled_prices.iloc[-1].values
    port_price = np.dot(best_weights, last_prices)
    periods = 12  # ۱۲ دوره (مثلاً ماهانه)
    simulated = np.cumprod(1 + np.random.normal(mu, sigma, (1000, periods)), axis=1) * port_price
    min_pred = simulated.min(axis=1).mean()
    max_pred = simulated.max(axis=1).mean()
    st.markdown(f"<div style='text-align:right;'>با توجه به بازده و ریسک پرتفو بهینه، <b>کمترین قیمت مورد انتظار پرتفو</b> در دوره آینده: <span style='color:red'>{min_pred:,.0f}</span><br> <b>بیشترین قیمت مورد انتظار پرتفو</b>: <span style='color:green'>{max_pred:,.0f}</span></div>", unsafe_allow_html=True)
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Box(y=simulated[:,-1], name="قیمت پایان دوره", boxmean=True, marker_color='blue'))
    fig_pred.add_trace(go.Scatter(y=[min_pred]*periods, x=list(range(1,periods+1)), mode="lines", name="کمترین قیمت میانگین", line=dict(color="red", dash="dot")))
    fig_pred.add_trace(go.Scatter(y=[max_pred]*periods, x=list(range(1,periods+1)), mode="lines", name="بیشترین قیمت میانگین", line=dict(color="green", dash="dot")))
    fig_pred.update_layout(title="پیش‌بینی قیمت پرتفو بهینه (شبیه‌سازی مونت‌کارلو)",
                          xaxis_title="تعداد دوره آینده", yaxis_title="قیمت پیش‌بینی‌شده", showlegend=True)
    st.plotly_chart(fig_pred, use_container_width=True)
    st.markdown("<div style='text-align:right;'>این نمودار توزیع انتهای قیمت پرتفو شبیه‌سازی‌شده را بر اساس بازده و ریسک فعلی نمایش می‌دهد. خطوط قرمز و سبز حداقل و حداکثر میانگین قیمت را نشان می‌دهند.</div>", unsafe_allow_html=True)

    # ==== توضیح کلی سبک‌ها ====
    st.markdown("""
    <div style='text-align:right;direction:rtl;'>
    <h4>راهنمای کوتاه سبک‌های بهینه‌سازی:</h4>
    <b>مرز کارا:</b> بهترین ترکیب بازده و ریسک.<br>
    <b>CVaR:</b> کمترین ریسک زیان شدید.<br>
    <b>برابری ریسک:</b> توزیع ریسک مساوی بین دارایی‌ها.<br>
    <b>خطای ردیابی:</b> کمترین فاصله از شاخص مبنا.<br>
    <b>نسبت اطلاعات:</b> ماکزیمم اختلاف بازده نسبت به ریسک اضافی.<br>
    <b>Kelly:</b> بیشترین رشد سرمایه.<br>
    <b>Sortino:</b> ماکزیمم سود نسبت به زیانهای پایین‌تر از هدف.<br>
    <b>Omega:</b> بیشترین نسبت سود به زیان.<br>
    <b>Drawdown:</b> کمترین افت سرمایه.<br>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("برای شروع، فایل‌های داده با ستون‌های 'Date' و 'Price' را آپلود کنید.")
