import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

import yfinance as yf

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو، CVaR و Married Put", layout="wide")
st.title("📊 ابزار تحلیل پرتفو با روش مونت‌کارلو، CVaR و استراتژی Married Put")

# راهنمای یاهو فاینانس (در ابتدای صفحه)
with st.expander("📘 راهنمای دریافت داده آنلاین از یاهو فاینانس"):
    st.markdown("""
    <div dir="rtl" style="text-align: right; font-size: 15px">
    <b>نحوه دانلود داده:</b><br>
    - نماد هر دارایی را مطابق سایت Yahoo Finance وارد کنید.<br>
    - نمادها را با <b>کاما</b> و <b>بدون فاصله</b> وارد کنید.<br>
    - مثال: <b>BTC-USD,AAPL,ETH-USD</b><br>
    - برای بیت‌کوین: <b>BTC-USD</b><br>
    - برای اپل: <b>AAPL</b><br>
    - برای اتریوم: <b>ETH-USD</b><br>
    - برای شاخص S&P500: <b>^GSPC</b><br>
    <br>
    <b>توضیحات بیشتر:</b><br>
    - نماد هر دارایی را می‌توانید در سایت <a href="https://finance.yahoo.com" target="_blank">Yahoo Finance</a> جستجو کنید.<br>
    - اگر چند نماد وارد می‌کنید، فقط با کاما جدا کنید و فاصله نگذارید.<br>
    - داده‌های دانلودشده دقیقاً مانند فایل CSV در ابزار استفاده می‌شوند.<br>
    </div>
    """, unsafe_allow_html=True)

def read_csv_file(file):
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip().str.lower().str.replace('%', '')
        df.rename(columns={'date': 'Date', 'price': 'Price'}, inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True, key="uploader"
)

period = st.sidebar.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'شش‌ماهه'])
resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'شش‌ماهه': '2Q'}[period]
annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'شش‌ماهه': 2}[period]
user_risk = st.sidebar.slider("ریسک هدف پرتفو (انحراف معیار سالانه)", 0.01, 1.0, 0.25, 0.01)
cvar_alpha = st.sidebar.slider("سطح اطمینان CVaR", 0.80, 0.99, 0.95, 0.01)

# === فیچر: دانلود داده آنلاین از یاهو فاینانس ===
with st.sidebar.expander("📥 دانلود داده آنلاین از یاهو فاینانس"):
    st.markdown("""
    <div dir="rtl" style="text-align: right; font-size: 14px">
    <b>راهنما:</b><br>
    - نماد هر دارایی را مطابق سایت یاهو فاینانس وارد کنید.<br>
    - نمادها را با کاما و بدون فاصله وارد کنید.<br>
    - مثال: <b>BTC-USD,AAPL,GOOGL,ETH-USD</b><br>
    - برای بیت‌کوین: <b>BTC-USD</b> <br>
    - برای اپل: <b>AAPL</b> <br>
    - برای اتریوم: <b>ETH-USD</b> <br>
    - برای شاخص S&P500: <b>^GSPC</b> <br>
    </div>
    """, unsafe_allow_html=True)
    tickers_input = st.text_input("نماد دارایی‌ها (با کاما و بدون فاصله)")
    start = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
    end = st.date_input("تاریخ پایان", value=pd.to_datetime("today"))
    download_btn = st.button("دریافت داده آنلاین")

downloaded_dfs = []
if download_btn and tickers_input.strip():
    tickers = [t.strip() for t in tickers_input.strip().split(",") if t.strip()]
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
        if not data.empty:
            for t in tickers:
                if len(tickers) == 1:
                    df = data.reset_index()[['Date', 'Close']].rename(columns={'Close': 'Price'})
                else:
                    if t in data.columns.levels[0]:
                        df_t = data[t].reset_index()[['Date', 'Close']].rename(columns={'Close': 'Price'})
                        df = df_t
                    else:
                        st.warning(f"داده‌ای برای نماد {t} دریافت نشد.")
                        continue
                df['Date'] = pd.to_datetime(df['Date'])
                downloaded_dfs.append((t, df))
                st.success(f"داده {t} با موفقیت دانلود شد.")
        else:
            st.error("داده‌ای دریافت نشد!")
    except Exception as ex:
        st.error(f"خطا در دریافت داده: {ex}")

# نمایش داده‌های دانلودشده (جدول)
if downloaded_dfs:
    st.markdown('<div dir="rtl" style="text-align: right;"><b>داده‌های دانلودشده از یاهو فاینانس:</b></div>', unsafe_allow_html=True)
    for t, df in downloaded_dfs:
        st.markdown(f"<div dir='rtl' style='text-align: right;'><b>{t}</b></div>", unsafe_allow_html=True)
        st.dataframe(df.head())

if uploaded_files or downloaded_dfs:
    prices_df = pd.DataFrame()
    asset_names = []
    insured_assets = {}

    # ابتدا داده‌های دانلودشده از یاهو
    for t, df in downloaded_dfs:
        name = t
        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"داده آنلاین {name} باید دارای ستون‌های 'Date' و 'Price' باشد.")
            continue
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]
        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

    # سپس فایل‌های آپلودی کاربر
    for file in uploaded_files:
        df = read_csv_file(file)
        if df is None:
            continue

        name = file.name.split('.')[0]
        if 'Date' not in df.columns or 'Price' not in df.columns:
            st.warning(f"فایل {name} باید دارای ستون‌های 'Date' و 'Price' باشد.")
            continue

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = df['Price'].astype(str).str.replace(',', '')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        df = df[['Date', 'Price']].set_index('Date')
        df.columns = [name]

        prices_df = df if prices_df.empty else prices_df.join(df, how='inner')
        asset_names.append(name)

        st.sidebar.markdown(f"---\n### ⚙️ تنظیمات بیمه برای دارایی: `{name}`")
        insured = st.sidebar.checkbox(f"📌 فعال‌سازی بیمه برای {name}", key=f"insured_{name}")
        if insured:
            loss_percent = st.sidebar.number_input(f"📉 درصد ضرر معامله پوت برای {name}", 0.0, 100.0, 30.0, step=0.01, key=f"loss_{name}")
            strike = st.sidebar.number_input(f"🎯 قیمت اعمال پوت برای {name}", 0.0, 1e6, 100.0, step=0.01, key=f"strike_{name}")
            premium = st.sidebar.number_input(f"💰 قیمت قرارداد پوت برای {name}", 0.0, 1e6, 5.0, step=0.01, key=f"premium_{name}")
            amount = st.sidebar.number_input(f"📦 مقدار قرارداد برای {name}", 0.0, 1e6, 1.0, step=0.01, key=f"amount_{name}")
            spot_price = st.sidebar.number_input(f"📌 قیمت فعلی دارایی پایه {name}", 0.0, 1e6, 100.0, step=0.01, key=f"spot_{name}")
            asset_amount = st.sidebar.number_input(f"📦 مقدار دارایی پایه {name}", 0.0, 1e6, 1.0, step=0.01, key=f"base_{name}")
            insured_assets[name] = {
                'loss_percent': loss_percent,
                'strike': strike,
                'premium': premium,
                'amount': amount,
                'spot': spot_price,
                'base': asset_amount
            }

    if prices_df.empty:
        st.error("❌ داده‌ی معتبری برای تحلیل یافت نشد.")
        st.stop()

    st.subheader("🧪 پیش‌نمایش داده‌ها")
    st.dataframe(prices_df.tail())

    resampled_prices = prices_df.resample(resample_rule).last().dropna()
    returns = resampled_prices.pct_change().dropna()
    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_devs = np.sqrt(np.diag(cov_matrix))

    adjusted_cov = cov_matrix.copy()
    preference_weights = []

    for i, name in enumerate(asset_names):
        if name in insured_assets:
            risk_scale = 1 - insured_assets[name]['loss_percent'] / 100
            adjusted_cov.iloc[i, :] *= risk_scale
            adjusted_cov.iloc[:, i] *= risk_scale
            preference_weights.append(1 / (std_devs[i] * risk_scale**0.7))  # وزن بیشتر در بیمه
        else:
            preference_weights.append(1 / std_devs[i])
    preference_weights = np.array(preference_weights)
    preference_weights /= np.sum(preference_weights)

    n_portfolios = 10000
    n_mc = 1000
    results = np.zeros((5 + len(asset_names), n_portfolios))
    np.random.seed(42)
    rf = 0

    downside = returns.copy()
    downside[downside > 0] = 0

    for i in range(n_portfolios):
        weights = np.random.random(len(asset_names)) * preference_weights
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(adjusted_cov, weights)))
        downside_risk = np.sqrt(np.dot(weights.T, np.dot(downside.cov() * annual_factor, weights)))
        sharpe_ratio = (port_return - rf) / port_std
        sortino_ratio = (port_return - rf) / downside_risk if downside_risk > 0 else np.nan

        mc_sims = np.random.multivariate_normal(mean_returns/annual_factor, adjusted_cov/annual_factor, n_mc)
        port_mc_returns = np.dot(mc_sims, weights)
        VaR = np.percentile(port_mc_returns, (1 - cvar_alpha) * 100)
        CVaR = port_mc_returns[port_mc_returns <= VaR].mean() if np.any(port_mc_returns <= VaR) else VaR

        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3, i] = sortino_ratio
        results[4, i] = -CVaR
        results[5:, i] = weights

    best_idx = np.argmin(np.abs(results[1] - user_risk))
    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_sortino = results[3, best_idx]
    best_weights = results[5:, best_idx]

    best_cvar_idx = np.argmin(results[4])
    best_cvar_return = results[0, best_cvar_idx]
    best_cvar_risk = results[1, best_cvar_idx]
    best_cvar_cvar = results[4, best_cvar_idx]
    best_cvar_weights = results[5:, best_cvar_idx]

    st.subheader("📊 داشبورد خلاصه پرتفو")
    total_weight = np.sum(best_weights)
    st.markdown(f'''
    <div dir="rtl" style="text-align: right">
    <b>بازده سالانه پرتفو:</b> {best_return:.2%}<br>
    <b>ریسک سالانه پرتفو:</b> {best_risk:.2%}<br>
    <b>نسبت شارپ:</b> {best_sharpe:.2f}<br>
    <b>بیشترین وزن:</b> {asset_names[np.argmax(best_weights)]} ({np.max(best_weights)*100:.2f}%)<br>
    <b>کمترین وزن:</b> {asset_names[np.argmin(best_weights)]} ({np.min(best_weights)*100:.2f}%)<br>
    </div>
    ''', unsafe_allow_html=True)
    fig_pie = go.Figure(data=[go.Pie(labels=asset_names, values=best_weights * 100, hole=.5, textinfo='label+percent')])
    fig_pie.update_layout(title="توزیع وزنی پرتفو بهینه")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("📈 پرتفو بهینه (مونت‌کارلو)")
    st.markdown(f"""
    - ✅ بازده سالانه: **{best_return:.2%}**
    - ⚠️ ریسک سالانه (انحراف معیار): **{best_risk:.2%}**
    - 🧠 نسبت شارپ: **{best_sharpe:.2f}**
    - 📉 نسبت سورتینو: **{best_sortino:.2f}**
    """)
    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 وزن {name}: {best_weights[i]*100:.2f}%")
    st.markdown('''
    <div dir="rtl" style="text-align: right">
    <b>پرتفو بهینه (مونت‌کارلو):</b><br>
    شبیه‌سازی Monte Carlo یکی از روش‌های قدرتمند برای بهینه‌سازی پرتفو بر اساس بازده و ریسک است. در این روش، با تولید تصادفی هزاران ترکیب از وزن دارایی‌ها و محاسبه بازده و ریسک هر ترکیب، ترکیبی پیدا می‌شود که نزدیک‌ترین ریسک را به ریسک هدف کاربر دارد. این رویکرد به شما کمک می‌کند تا ترکیب دارایی‌هایی با بیشترین احتمال بازده و کمترین ریسک را شناسایی کنید. مزیت اصلی این روش انعطاف‌پذیری بالا و بررسی حالات مختلف بازار است.
    </div>
    ''', unsafe_allow_html=True)

    st.subheader(f"🟢 پرتفو بهینه بر اساس CVaR ({int(cvar_alpha*100)}%)")
    st.markdown(f"""
    - ✅ بازده سالانه: **{best_cvar_return:.2%}**
    - ⚠️ ریسک سالانه (انحراف معیار): **{best_cvar_risk:.2%}**
    - 🟠 CVaR ({int(cvar_alpha*100)}%): **{best_cvar_cvar:.2%}**
    """)
    for i, name in enumerate(asset_names):
        st.markdown(f"🔸 وزن {name}: {best_cvar_weights[i]*100:.2f}%")
    st.markdown(f'''
    <div dir="rtl" style="text-align: right">
    <b>پرتفو بهینه بر اساس CVaR ({int(cvar_alpha*100)}%):</b><br>
    معیار <b>CVaR</b> (ارزش در معرض ریسک شرطی) ریسک پرتفو را در شرایط بحرانی‌تر بازار اندازه‌گیری می‌کند. این معیار بهترین ابزار برای ارزیابی زیان‌های شدید و کم‌احتمال است. پرتفو بهینه بر اساس CVaR ترکیبی از دارایی‌ها را ارائه می‌دهد که در بدترین سناریوهای بازار، کمترین زیان ممکن را متحمل می‌شود. این سبک برای سرمایه‌گذارانی مناسب است که مدیریت ریسک حداکثری برایشان اهمیت دارد.
    </div>
    ''', unsafe_allow_html=True)

    st.subheader("📋 جدول مقایسه وزن دارایی‌ها (مونت‌کارلو و CVaR)")
    compare_df = pd.DataFrame({
        'دارایی': asset_names,
        'وزن مونت‌کارلو (%)': best_weights * 100,
        f'وزن CVaR ({int(cvar_alpha*100)}%) (%)': best_cvar_weights * 100
    })
    compare_df['اختلاف وزن (%)'] = compare_df[f'وزن CVaR ({int(cvar_alpha*100)}%) (%)'] - compare_df['وزن مونت‌کارلو (%)']
    st.dataframe(compare_df.set_index('دارایی'), use_container_width=True, height=300)
    st.markdown('''
    <div dir="rtl" style="text-align: right">
    این جدول تفاوت وزن دارایی‌ها را در دو مدل بهینه‌سازی (مونت‌کارلو و CVaR) نشان می‌دهد. اختلاف وزن‌ها می‌تواند بیانگر میزان اهمیت مدیریت ریسک در هر دارایی باشد.
    </div>
    ''', unsafe_allow_html=True)

    fig_w = go.Figure()
    fig_w.add_trace(go.Bar(x=asset_names, y=best_weights*100, name='مونت‌کارلو'))
    fig_w.add_trace(go.Bar(x=asset_names, y=best_cvar_weights*100, name=f'CVaR {int(cvar_alpha*100)}%'))
    fig_w.update_layout(barmode='group', title="مقایسه وزن دارایی‌ها در دو سبک")
    st.plotly_chart(fig_w, use_container_width=True)
    st.markdown('''
    <div dir="rtl" style="text-align: right">
    این نمودار بصری به مقایسه وزن‌های بهینه هر دارایی در هر یک از دو روش می‌پردازد و به شما کمک می‌کند تاثیر مدل انتخابی بر سبد دارایی خود را بهتر درک کنید.
    </div>
    ''', unsafe_allow_html=True)

    st.subheader("🌈 نمودار مرز کارا")
    fig = px.scatter(
        x=results[1]*100,
        y=results[0]*100,
        color=results[2],
        labels={'x': 'ریسک (%)', 'y': 'بازده (%)'},
        title='پرتفوهای شبیه‌سازی‌شده (مونت‌کارلو) و مرز CVaR',
        color_continuous_scale='Viridis'
    )
    fig.add_trace(go.Scatter(x=[best_risk*100], y=[best_return*100],
                             mode='markers', marker=dict(size=12, color='red', symbol='star'),
                             name='پرتفوی بهینه مونت‌کارلو'))
    fig.add_trace(go.Scatter(x=[best_cvar_risk*100], y=[best_cvar_return*100],
                             mode='markers', marker=dict(size=12, color='orange', symbol='star'),
                             name='پرتفوی بهینه CVaR'))
    cvar_sorted_idx = np.argsort(results[4])
    fig.add_trace(go.Scatter(
        x=results[1, cvar_sorted_idx]*100,
        y=results[0, cvar_sorted_idx]*100,
        mode='lines',
        line=dict(color='orange', dash='dot'),
        name='مرز کارا (CVaR)'
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('''
    <div dir="rtl" style="text-align: right">
    <b>مرز کارا</b> (Efficient Frontier) مجموعه‌ای از پرتفوهاست که بهترین بازده ممکن را برای سطح مشخصی از ریسک ارائه می‌دهد. این نمودار به شما کمک می‌کند پرتفوهای مختلف را از نظر ریسک و بازده و همچنین تاثیر مدیریت ریسک با CVaR مقایسه کنید.
    </div>
    ''', unsafe_allow_html=True)

    st.subheader("🔵 نمودار بازده- CVaR برای پرتفوها")
    fig_cvar = px.scatter(
        x=results[4], y=results[0],
        labels={'x': f'CVaR ({int(cvar_alpha*100)}%)', 'y': 'بازده'},
        title='پرتفوها بر اساس بازده و CVaR',
        color=results[1], color_continuous_scale='Blues'
    )
    fig_cvar.add_trace(go.Scatter(x=[best_cvar_cvar], y=[best_cvar_return],
                                  mode='markers', marker=dict(size=12, color='red', symbol='star'),
                                  name='پرتفوی بهینه CVaR'))
    st.plotly_chart(fig_cvar, use_container_width=True)
    st.markdown('''
    <div dir="rtl" style="text-align: right">
    این نمودار رابطه بین بازده و معیار ریسک <b>CVaR</b> را در سبدهای مختلف نمایش می‌دهد. هرچه مقدار CVaR کمتر باشد، پرتفو در سناریوهای بحرانی کمتر آسیب‌پذیر است.
    </div>
    ''', unsafe_allow_html=True)

    st.subheader("💡 دارایی‌های پیشنهادی بر اساس نسبت بازده به ریسک")
    asset_scores = {}
    for i, name in enumerate(asset_names):
        insured_factor = 1 - insured_assets.get(name, {}).get('loss_percent', 0)/100 if name in insured_assets else 1
        score = mean_returns[i] / (std_devs[i]*insured_factor)
        asset_scores[name] = score

    sorted_assets = sorted(asset_scores.items(), key=lambda x: x[1], reverse=True)
    st.markdown("**به ترتیب اولویت:**")
    for name, score in sorted_assets:
        insured_str = " (بیمه شده)" if name in insured_assets else ""
        st.markdown(f"🔸 **{name}{insured_str}** | نسبت بازده به ریسک: {score:.2f}")

    st.markdown('''
    <div dir="rtl" style="text-align: right">
    این بخش دارایی‌هایی را که بالاترین نسبت بازده به ریسک را دارند معرفی می‌کند. دارایی بیمه‌شده دارای ریسک تعدیل‌شده است که با علامت مشخص شده‌اند.
    </div>
    ''', unsafe_allow_html=True)

    for name, info in insured_assets.items():
        st.subheader(f"📉 نمودار سود و زیان استراتژی Married Put - {name}")
        x = np.linspace(info['spot'] * 0.5, info['spot'] * 1.5, 200)
        asset_pnl = (x - info['spot']) * info['base']
        put_pnl = np.where(x < info['strike'], (info['strike'] - x) * info['amount'], 0) - info['premium'] * info['amount']
        total_pnl = asset_pnl + put_pnl

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=x[total_pnl>=0], y=total_pnl[total_pnl>=0], mode='lines', name='سود', line=dict(color='green', width=3)
        ))
        fig2.add_trace(go.Scatter(
            x=x[total_pnl<0], y=total_pnl[total_pnl<0], mode='lines', name='زیان', line=dict(color='red', width=3)
        ))
        fig2.add_trace(go.Scatter(
            x=x, y=asset_pnl, mode='lines', name='دارایی پایه', line=dict(dash='dot', color='gray')
        ))
        fig2.add_trace(go.Scatter(
            x=x, y=put_pnl, mode='lines', name='پوت', line=dict(dash='dot', color='blue')
        ))
        zero_crossings = np.where(np.diff(np.sign(total_pnl)))[0]
        if len(zero_crossings):
            breakeven_x = x[zero_crossings[0]]
            fig2.add_trace(go.Scatter(x=[breakeven_x], y=[0], mode='markers+text', marker=dict(color='orange', size=10),
                                      text=["سر به سر"], textposition="bottom center", name='سر به سر'))
        max_pnl = np.max(total_pnl)
        max_x = x[np.argmax(total_pnl)]
        fig2.add_trace(go.Scatter(x=[max_x], y=[max_pnl], mode='markers+text', marker=dict(color='green', size=10),
                                  text=[f"{(max_pnl/(info['spot']*info['base'])*100):.1f}% سود"], textposition="top right",
                                  showlegend=False))
        fig2.update_layout(title='نمودار سود و زیان', xaxis_title='قیمت دارایی در سررسید', yaxis_title='سود/زیان')
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown('''
        <div dir="rtl" style="text-align: right">
        <b>استراتژی Married Put:</b><br>
        در این استراتژی، سرمایه‌گذار همزمان با خرید دارایی پایه، یک قرارداد اختیار فروش (Put) نیز خریداری می‌کند. این کار باعث می‌شود در صورت کاهش شدید قیمت دارایی، زیان‌های احتمالی پوشش داده شود، اما در صورت رشد قیمت، همچنان از افزایش قیمت بهره‌مند شوید. بنابراین، Married Put ابزاری برای بیمه کردن سبد دارایی در برابر سقوط‌های غیرمنتظره است.
        </div>
        ''', unsafe_allow_html=True)

        if st.button(f"📷 ذخیره نمودار Married Put برای {name}"):
            try:
                img_bytes = fig2.to_image(format="png")
                st.download_button("دانلود تصویر", img_bytes, file_name=f"married_put_{name}.png")
            except Exception as e:
                st.error(f"❌ خطا در ذخیره تصویر: {e}")

    st.subheader("🔮 پیش‌بینی قیمت و بازده آتی هر دارایی")
    future_months = 6 if period == 'شش‌ماهه' else (3 if period == 'سه‌ماهه' else 1)
    for i, name in enumerate(asset_names):
        last_price = resampled_prices[name].iloc[-1]
        mu = mean_returns[i] / annual_factor
        sigma = std_devs[i] / np.sqrt(annual_factor)
        sim_prices = []
        n_sim = 500
        for _ in range(n_sim):
            sim = last_price * np.exp(np.cumsum(np.random.normal(mu, sigma, future_months)))
            sim_prices.append(sim[-1])
        sim_prices = np.array(sim_prices)
        future_price_mean = np.mean(sim_prices)
        future_return = (future_price_mean - last_price) / last_price

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=sim_prices, nbinsx=20, name="پیش‌بینی قیمت", marker_color='purple'))
        fig3.add_vline(x=future_price_mean, line_dash="dash", line_color="green", annotation_text=f"میانگین: {future_price_mean:.2f}")
        fig3.update_layout(title=f"پیش‌بینی قیمت {name} در {future_months} {'ماه' if future_months>1 else 'ماه'} آینده",
            xaxis_title="قیمت انتهایی", yaxis_title="تعداد شبیه‌سازی")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f"📈 **میانگین قیمت آینده:** {future_price_mean:.2f} | 📊 **درصد بازده آتی:** {future_return:.2%}")

    st.markdown('''
    <div dir="rtl" style="text-align: right">
    <b>پیش‌بینی قیمت آینده با شبیه‌سازی تصادفی:</b><br>
    در این مدل، با استفاده از مدل‌سازی تصادفی (Random Walk) و پارامترهای بازده و ریسک تاریخی، قیمت‌های آتی هر دارایی در بازه زمانی مشخص شبیه‌سازی می‌شود. این پیش‌بینی به شما دید بهتری نسبت به سناریوهای احتمالی قیمت در آینده می‌دهد، اما باید توجه داشت که پیش‌بینی‌ها هیچ‌گاه قطعی نیستند و فقط جنبه تحلیلی و آماری دارند.
    </div>
    ''', unsafe_allow_html=True)

else:
    st.warning("⚠️ لطفاً فایل‌های CSV شامل ستون‌های Date و Price را آپلود کنید یا از بخش دانلود آنلاین داده استفاده کنید.")
