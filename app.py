import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="تحلیل پرتفو و بیمه آپشن", layout="wide")
st.markdown("<h1 style='text-align: right;'>📊 ابزار جامع تحلیل پرتفو با بیمه مرید پوت و پریوت پوت</h1>", unsafe_allow_html=True)

# ۱. بازه زمانی تحلیل
st.sidebar.markdown("<h3 style='text-align: right;'>۱. بازه زمانی تحلیل</h3>", unsafe_allow_html=True)
date_range_mode = st.sidebar.radio("انتخاب بازه:", ["کل داده", "بازه دلخواه"], horizontal=True)
date_start, date_end = None, None
if date_range_mode == "بازه دلخواه":
    date_start = st.sidebar.date_input("تاریخ شروع", value=datetime(2022,1,1))
    date_end = st.sidebar.date_input("تاریخ پایان", value=datetime.now())
    if date_end < date_start:
        st.sidebar.error("تاریخ پایان باید بعد از تاریخ شروع باشد.")

# ۲. بارگذاری و تنظیم دارایی‌ها و بیمه
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
        # بیمه
        insure = st.sidebar.checkbox(f"فعالسازی بیمه برای {asset_name}", key=f"insure_{asset_name}")
        if insure:
            strat_type = st.sidebar.selectbox(f"نوع بیمه {asset_name}", ["مرید پوت (Protective Put)", "پریوت پوت (Perpetual Put)"], key=f"stype_{asset_name}")
            entry_price = st.sidebar.number_input(f"قیمت خرید دارایی پایه ({asset_name})", min_value=0.00001, max_value=100.0, value=1.0, step=0.01, key=f"entry_{asset_name}")
            strike = st.sidebar.number_input(f"قیمت اعمال (استرایک) ({asset_name})", min_value=0.00001, max_value=100.0, value=0.95, step=0.01, key=f"strike_{asset_name}")
            premium = st.sidebar.number_input(f"پریمیوم (حق بیمه) ({asset_name})", min_value=0.00001, max_value=100.0, value=0.01, step=0.01, key=f"premium_{asset_name}")
            pos_size = st.sidebar.number_input(f"مقدار دارایی پایه ({asset_name})", min_value=0.00001, max_value=100.0, value=1.0, step=0.01, key=f"possize_{asset_name}")
            opt_size = st.sidebar.number_input(f"مقدار قرارداد آپشن ({asset_name})", min_value=0.00001, max_value=100.0, value=1.0, step=0.01, key=f"optsize_{asset_name}")
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

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: right;'>۳. تنظیمات بازه بازده</h3>", unsafe_allow_html=True)
period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]

# ۴. انتخاب روش بهینه‌سازی
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

# استراتژی مرید پوت و پریوت پوت
def protective_put_payoff(price, entry, strike, premium, pos_size, opt_size):
    # (مقدار دارایی × سود/زیان دارایی) + (مقدار قرارداد × سود آپشن) - هزینه پریمیوم
    pl_stock = (price - entry) * pos_size
    pl_option = np.maximum(strike - price, 0) * opt_size
    cost = premium * opt_size
    total_pl = pl_stock + pl_option - cost
    pct = total_pl / (entry * pos_size)
    return pct

def perpetual_put_payoff(price, entry, strike, premium, pos_size, opt_size):
    # مشابه مرید پوت، اما بیمه همواره تمدید می‌شود (پریمیوم هر دوره کسر می‌شود)
    pl_stock = (price - entry) * pos_size
    pl_option = np.maximum(strike - price, 0) * opt_size
    # فرض: هزینه پریمیوم برای تعداد دوره‌ها طبق اختلاف قیمت تا پایان
    n_periods = np.maximum(np.floor((entry - price) / (entry - strike)), 1)
    cost = premium * opt_size * n_periods
    total_pl = pl_stock + pl_option - cost
    pct = total_pl / (entry * pos_size)
    return pct

# اعمال بیمه روی بازده دارایی‌ها
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

    # شبیه‌سازی پرتفو
    results = []
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

    # انتخاب پرتفو بهینه
    if method == "واریانس میانگین (مرز کارا)":
        best = max(results, key=lambda x: x["return"] / x["risk"])
        explain = "پرتفو با بیشترین نسبت بازده به ریسک (Sharpe) روی مرز کارا."
    elif method == "کمینه CVaR (ارزش در معرض ریسک مشروط)":
        best = min(results, key=lambda x: x["cvar"])
        explain = "پرتفو با کمترین مقدار CVaR (خطر دنباله)."
    elif method == "برابری ریسک":
        best = min(results, key=lambda x: x["risk_budget"])
        explain = "پرتفو با نزدیک‌ترین سهم ریسک مساوی بین دارایی‌ها."
    elif method == "حداقل خطای ردیابی":
        best = min(results, key=lambda x: x["tracking"])
        explain = "پرتفو با کمترین خطای ردیابی نسبت به معیار پرتفو (میانگین دارایی‌ها)."
    elif method == "بیشینه نسبت اطلاعات":
        best = max(results, key=lambda x: x["info"])
        explain = "پرتفو با بیشترین نسبت اطلاعات نسبت به معیار پرتفو."
    elif method == "بیشینه نرخ رشد هندسی (Kelly)":
        best = max(results, key=lambda x: x["kelly"])
        explain = "پرتفو با بیشترین رشد هندسی مورد انتظار (معیار Kelly)."
    elif method == "بیشینه نسبت سورتینو":
        best = max(results, key=lambda x: x["sortino"])
        explain = "پرتفو با بیشترین نسبت سورتینو با حداقل بازده مقبول."
    elif method == "بیشینه نسبت امگا":
        best = max(results, key=lambda x: x["omega"])
        explain = "پرتفو با بیشترین نسبت امگا (فراتر از بازده مقبول)."
    elif method == "کمترین افت سرمایه":
        best = max(results, key=lambda x: x["drawdown"])
        explain = "پرتفو با کمترین افت سرمایه (حداقل Drawdown)."
    elif method == "Black-Litterman":
        best = max(results, key=lambda x: x["return"] / x["risk"])
        explain = "مدل Black-Litterman (در این نسخه: مرز کارا بدون دیدگاه خاص)."
    else:
        best = results[0]
        explain = "-"

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

    st.markdown("<h3 style='text-align:right;'>📌 نمودار توزیع وزن پرتفو بهینه</h3>", unsafe_allow_html=True)
    fig = go.Figure(data=[go.Pie(labels=asset_names, values=best['weights'], hole=0.5)])
    fig.update_layout(title="توزیع وزن دارایی‌ها")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<h3 style='text-align:right;'>🌈 نمودار مرز کارا شبیه‌سازی‌شده</h3>", unsafe_allow_html=True)
    df = pd.DataFrame(results)
    fig2 = px.scatter(df, x="risk", y="return", color="sortino",
                      hover_data=["info", "omega", "cvar"], title="مرز کارا با رنگ‌بندی نسبت سورتینو")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<h3 style='text-align:right;'>📋 جدول وزن پرتفو بهینه</h3>", unsafe_allow_html=True)
    table = pd.DataFrame({"دارایی": asset_names, "وزن بهینه (%)": np.array(best['weights'])*100,
                          "حداقل وزن (%)": [asset_settings[n]["min"]*100 for n in asset_names],
                          "حداکثر وزن (%)": [asset_settings[n]["max"]*100 for n in asset_names]})
    st.dataframe(table.set_index("دارایی"), use_container_width=True, height=300)

    # نمودار استراتژی بیمه برای هر دارایی
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
            # سربه‌سر تقریبی
            idx_cross = np.argmin(np.abs(profit))
            breakeven = price_range[idx_cross]
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=price_range, y=profit*100, mode="lines",
                line=dict(color="green"),
                name="سود/زیان استراتژی"
            ))
            # ناحیه سود سبز
            green_x = price_range[profit >= 0]
            green_y = profit[profit >= 0]*100
            if len(green_x) > 0:
                fig3.add_trace(go.Scatter(
                    x=green_x, y=green_y, mode='lines',
                    fill='tozeroy', fillcolor='rgba(0,255,0,0.15)', line=dict(color="rgba(0,0,0,0)"),
                    name="ناحیه سود"
                ))
            # ناحیه زیان قرمز
            red_x = price_range[profit < 0]
            red_y = profit[profit < 0]*100
            if len(red_x) > 0:
                fig3.add_trace(go.Scatter(
                    x=red_x, y=red_y, mode='lines',
                    fill='tozeroy', fillcolor='rgba(255,0,0,0.15)', line=dict(color="rgba(0,0,0,0)"),
                    name="ناحیه زیان"
                ))
            # خط سربه‌سر
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
                <b>ویژگی نمودار:</b><br>
                - محور افقی: قیمت دارایی پایه<br>
                - محور عمودی: درصد سود/زیان<br>
                - ناحیه سبز: سود (سربه‌سر به بالا)<br>
                - ناحیه قرمز: زیان (سربه‌سر به پایین)<br>
                - خط آبی: نقطه سربه‌سر<br>
                - سود/زیان لحظه‌ای و درصد نمایش داده می‌شود<br>
                </div>""", unsafe_allow_html=True)

    # توضیح فارسی استراتژی‌ها
    st.markdown(
        """
        <div style='text-align:right;direction:rtl;'>
        <h3>راهنمای استراتژی بیمه:</h3>
        <b>مرید پوت (Protective Put):</b> خرید دارایی و خرید اختیار فروش با قیمت اعمال مشخص. اگر قیمت پایین بیاید، اختیار فروش ضرر را محدود می‌کند و ریسک را کاهش می‌دهد.<br>
        <b>پریوت پوت (Perpetual Put):</b> مشابه مرید پوت است اما اختیار فروش به صورت دائمی تمدید می‌شود و همیشه بیمه فعال است.<br>
        <br>
        پارامترهای هر دارایی:<br>
        - قیمت خرید دارایی پایه<br>
        - قیمت اعمال اختیار فروش (استرایک)<br>
        - پریمیوم (حق بیمه)<br>
        - مقدار دارایی پایه<br>
        - مقدار قرارداد آپشن<br>
        <br>
        بازه مقادیر مجاز: 0.00001 تا 100 <br>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("برای شروع، فایل‌های داده با ستون‌های 'Date' و 'Price' را آپلود کنید.")
