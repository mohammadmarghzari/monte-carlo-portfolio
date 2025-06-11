import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from collections import Counter
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="شبیه‌ساز پرتفوی حرفه‌ای", layout="wide")

# ---------- Custom CSS for Advanced UI ----------
st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 5px 15px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stNumberInput>input {
        background-color: #2e2e2e;
        color: #e0e0e0;
        border: 1px solid #444;
        border-radius: 5px;
    }
    .stExpander {
        background-color: #2e2e2e;
        border: 1px solid #444;
        border-radius: 5px;
    }
    .stExpander > div[data-testid="stExpanderHeader"] {
        color: #e0e0e0;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2e2e2e;
        border-bottom: 1px solid #444;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2e2e2e;
        color: #bbb;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3e3e3e;
    }
    .stTabs [data-baseweb="tab--selected"] {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px 5px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helper Functions ----------

def format_money(val):
    if val == 0:
        return "۰ دلار"
    elif val >= 1:
        return "{:,.0f} دلار".format(val)
    else:
        return "{:.3f} دلار".format(val).replace('.', '٫')

def format_percent(val):
    return "{:.3f}%".format(val*100).replace('.', '٫')

def format_float(val):
    if abs(val) >= 1:
        return "{:,.3f}".format(val).rstrip('0').rstrip('.')
    else:
        return "{:.6f}".format(val).rstrip('0').rstrip('.')

def read_csv_file(file):
    try:
        file.seek(0)
        df_try = pd.read_csv(file)
        cols_lower = [str(c).strip().lower() for c in df_try.columns]
        if any(x in cols_lower for x in ['date']):
            df = df_try.copy()
        else:
            file.seek(0)
            df = pd.read_csv(file, header=None)
            header_idx = None
            for i in range(min(5, len(df))):
                row = [str(x).strip().lower() for x in df.iloc[i].tolist()]
                if any('date' == x for x in row):
                    header_idx = i
                    break
            if header_idx is None:
                raise Exception("سطر عنوان مناسب (شامل date) یافت نشد.")
            header_row = df.iloc[header_idx].tolist()
            df = df.iloc[header_idx+1:].reset_index(drop=True)
            df.columns = header_row

        date_col = [c for c in df.columns if str(c).strip().lower() == 'date']
        if not date_col:
            raise Exception("ستون تاریخ با نام 'Date' یا مشابه آن یافت نشد.")
        date_col = date_col[0]
        price_candidates = [c for c in df.columns if str(c).strip().lower() in ['price', 'close', 'adj close', 'open']]
        if not price_candidates:
            price_candidates = [c for c in df.columns if c != date_col]
        if not price_candidates:
            raise Exception("ستون قیمت مناسب یافت نشد.")
        price_col = price_candidates[0]
        df = df[[date_col, price_col]].dropna()
        if df.empty:
            raise Exception("پس از حذف داده‌های خالی، داده‌ای باقی نماند.")

        df = df.rename(columns={date_col: "Date", price_col: "Price"})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        if df.empty:
            raise Exception("پس از تبدیل نوع داده، داده معتبری باقی نماند.")
        return df
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

def get_price_dataframe_from_yf(data, t):
    if isinstance(data.columns, pd.MultiIndex):
        if t in data.columns.levels[0]:
            df_t = data[t].reset_index()
            price_col = None
            for col in ['Close', 'Adj Close', 'Open']:
                if col in df_t.columns:
                    price_col = col
                    break
            if price_col is None:
                return None, f"هیچ یک از ستون‌های قیمت (Close, Adj Close, Open) برای {t} پیدا نشد."
            df = df_t[['Date', price_col]].rename(columns={price_col: 'Price'})
            return df, None
        else:
            return None, f"نماد {t} در داده‌های دریافتی وجود ندارد."
    else:
        if 'Date' not in data.columns:
            data = data.reset_index()
        price_col = None
        for col in ['Close', 'Adj Close', 'Open']:
            if col in data.columns:
                price_col = col
                break
        if price_col is None:
            return None, f"هیچ یک از ستون‌های قیمت (Close, Adj Close, Open) برای {t} پیدا نشد."
        df = data[['Date', price_col]].rename(columns={price_col: 'Price'})
        return df, None

def calc_option_return(row_type, price, prev_price, strike, premium, qty):
    if row_type == 'خرید دارایی':
        return (price - prev_price) / prev_price if prev_price != 0 else 0
    elif row_type == 'فروش دارایی':
        return (prev_price - price) / prev_price if prev_price != 0 else 0
    elif row_type == 'خرید کال':
        return (max(price - strike, 0) - premium) / prev_price if prev_price != 0 else 0
    elif row_type == 'فروش کال':
        return (premium - max(price - strike, 0)) / prev_price if prev_price != 0 else 0
    elif row_type == 'خرید پوت':
        return (max(strike - price, 0) - premium) / prev_price if prev_price != 0 else 0
    elif row_type == 'فروش پوت':
        return (premium - max(strike - price, 0)) / prev_price if prev_price != 0 else 0
    elif row_type == 'فروش فیوچرز':
        return (prev_price - price) / prev_price if prev_price != 0 else 0
    else:
        return 0

def calc_options_series(option_rows, prices: pd.Series):
    rets = pd.Series(np.zeros(len(prices)), index=prices.index)
    prev_price = prices.iloc[0]
    for i in range(1, len(prices)):
        price = prices.iloc[i]
        ret_row = 0
        for row in option_rows:
            row_type, strike, premium, qty = row
            ret_row += qty * calc_option_return(row_type, price, prev_price, strike, premium, 1)
        rets.iloc[i] = ret_row
        prev_price = price
    return rets

def calculate_payoff(option_rows, current_price, purchase_price, price_range):
    payoffs = []
    total_premium = sum(qty * premium * purchase_price for _, _, premium, qty in option_rows if premium != 0)
    for price in price_range:
        total_payoff = 0
        for row_type, strike, premium, qty in option_rows:
            if row_type == 'خرید دارایی':
                total_payoff += qty * (price - purchase_price)
            elif row_type == 'فروش دارایی':
                total_payoff += qty * (purchase_price - price)
            elif row_type == 'خرید کال':
                total_payoff += qty * (max(price - strike, 0) - premium * purchase_price)
            elif row_type == 'فروش کال':
                total_payoff += qty * (premium * purchase_price - max(price - strike, 0))
            elif row_type == 'خرید پوت':
                total_payoff += qty * (max(strike - price, 0) - premium * purchase_price)
            elif row_type == 'فروش پوت':
                total_payoff += qty * (premium * purchase_price - max(strike - price, 0))
            elif row_type == 'فروش فیوچرز':
                total_payoff += qty * (purchase_price - price)
        payoffs.append(total_payoff - total_premium)
    return payoffs

def calculate_breakeven(option_rows, purchase_price):
    total_premium = sum(qty * premium * purchase_price for _, _, premium, qty in option_rows if premium != 0)
    for row_type, strike, premium, qty in option_rows:
        if row_type in ['خرید پوت', 'خرید کال']:
            return strike + (total_premium / qty) if qty != 0 else purchase_price
        elif row_type in ['فروش پوت', 'فروش کال']:
            return strike - (total_premium / qty) if qty != 0 else purchase_price
    return purchase_price + total_premium

def calculate_profit_loss_percent(payoffs, purchase_price, investment):
    return [(p / investment) * 100 for p in payoffs]

def sharpe_ratio(returns, risk_free=0, ann_factor=12):
    excess_ret = returns - risk_free/ann_factor
    mean = np.mean(excess_ret)
    std = np.std(excess_ret, ddof=1)
    if std == 0: return 0
    return (mean / std) * np.sqrt(ann_factor)

def sortino_ratio(returns, risk_free=0, ann_factor=12):
    excess_ret = returns - risk_free/ann_factor
    mean = np.mean(excess_ret)
    neg_returns = excess_ret[excess_ret < 0]
    downside_std = np.std(neg_returns, ddof=1) if len(neg_returns) > 0 else 0.0001
    return (mean / downside_std) * np.sqrt(ann_factor)

def annual_volatility(returns, ann_factor=12):
    return np.std(returns, ddof=1) * np.sqrt(ann_factor)

def annual_return(returns, ann_factor=12):
    compounded = np.prod(1 + returns) ** (ann_factor / len(returns)) - 1
    return compounded

def max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return np.min(drawdown)

def var(returns, alpha=0.95):
    return np.percentile(returns, (1-alpha)*100)

def cvar(returns, alpha=0.95):
    var_value = var(returns, alpha)
    return returns[returns <= var_value].mean() if np.any(returns <= var_value) else var_value

def efficient_frontier(mean_returns, cov_matrix, points=200, min_weights=None, max_weights=None):
    num_assets = len(mean_returns)
    results = np.zeros((3, points))
    weight_record = []
    for i in range(points):
        for _ in range(100):
            w = np.random.dirichlet(np.ones(num_assets), size=1)[0]
            if min_weights is not None:
                if not np.all(w >= min_weights): continue
            if max_weights is not None:
                if not np.all(w <= max_weights): continue
            break
        weights = w
        port_return = np.dot(weights, mean_returns)
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0,i] = port_std
        results[1,i] = port_return
        results[2,i] = (port_return) / port_std if port_std > 0 else 0
        weight_record.append(weights)
    return results, np.array(weight_record)

def calc_asset_stats(prices: pd.Series, freq='M', risk_free=0):
    if freq == 'M':
        returns = prices.resample('M').last().pct_change().dropna()
        ann_factor = 12
    elif freq == 'D':
        returns = prices.pct_change().dropna()
        ann_factor = 252
    elif freq == 'W':
        returns = prices.resample('W').last().pct_change().dropna()
        ann_factor = 52
    else:
        returns = prices.resample(freq).last().pct_change().dropna()
        ann_factor = 12

    sharpe = sharpe_ratio(returns, risk_free, ann_factor)
    sortino = sortino_ratio(returns, risk_free, ann_factor)
    volatility = annual_volatility(returns, ann_factor)
    total_return = annual_return(returns, ann_factor)
    implied_vol = np.std(returns, ddof=1) * np.sqrt(ann_factor)
    maxdd = max_drawdown(returns)
    mean_ann = np.mean(returns) * ann_factor
    mean_month = np.mean(returns)
    std_ann = np.std(returns, ddof=1) * np.sqrt(ann_factor)
    std_month = np.std(returns, ddof=1)
    min_ann = np.min(returns) * ann_factor
    max_ann = np.max(returns) * ann_factor
    min_month = np.min(returns)
    max_month = np.max(returns)
    var_95 = var(returns, 0.95)
    cvar_95 = cvar(returns, 0.95)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "volatility_ann": volatility,
        "total_return_ann": total_return,
        "implied_vol": implied_vol,
        "mean_ann": mean_ann,
        "mean_month": mean_month,
        "std_ann": std_ann,
        "std_month": std_month,
        "min_ann": min_ann,
        "max_ann": max_ann,
        "min_month": min_month,
        "max_month": max_month,
        "max_drawdown": maxdd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "returns": returns
    }

def forecast_prices(prices: pd.Series, horizons=[1, 3, 6]):
    returns = prices.pct_change().dropna()
    forecasts = {}
    for horizon in horizons:
        try:
            # ARIMA for trend
            arima_model = ARIMA(returns, order=(1,1,1))
            arima_fit = arima_model.fit()
            arima_forecast = arima_fit.forecast(steps=horizon)
            # GARCH for volatility
            garch_model = arch_model(returns, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            garch_forecast = garch_fit.forecast(horizon=horizon)
            variance = garch_forecast.variance.values[-1]
            # Convert returns to prices
            last_price = prices.iloc[-1]
            forecast_price = last_price * np.prod(1 + arima_forecast)
            conf_int = [
                forecast_price * np.exp(-1.96 * np.sqrt(variance.sum())),
                forecast_price * np.exp(1.96 * np.sqrt(variance.sum()))
            ]
            forecasts[horizon] = {
                'price': forecast_price,
                'conf_int': conf_int
            }
        except:
            forecasts[horizon] = {'price': prices.iloc[-1], 'conf_int': [prices.iloc[-1]*0.9, prices.iloc[-1]*1.1]}
    return forecasts

# ---------- Session State ----------
if "downloaded_dfs" not in st.session_state:
    st.session_state["downloaded_dfs"] = []
if "uploaded_dfs" not in st.session_state:
    st.session_state["uploaded_dfs"] = []
if "option_rows" not in st.session_state:
    st.session_state["option_rows"] = {}
if "investment_amount" not in st.session_state:
    st.session_state["investment_amount"] = 10000.0
if "hedge_status" not in st.session_state:
    st.session_state["hedge_status"] = {}

# ---------- Sidebar: Uploads and Download ----------
st.sidebar.header("📂 مدیریت داده‌ها")
with st.sidebar.expander("بارگذاری فایل‌ها 📤"):
    uploaded_files = st.file_uploader(
        "آپلود فایل‌های CSV (هر دارایی یک فایل)", type=['csv'], accept_multiple_files=True, key="uploader"
    )
    if uploaded_files:
        for file in uploaded_files:
            if not hasattr(file, "uploaded_in_session") or not file.uploaded_in_session:
                df = read_csv_file(file)
                if df is not None:
                    st.session_state["uploaded_dfs"].append((file.name.split('.')[0], df))
                file.uploaded_in_session = True

with st.sidebar.expander("دریافت داده آنلاین 📥"):
    st.markdown("""
    <div dir="rtl" style="text-align: right;">
    <b>راهنما:</b>
    <br>نمادها را با کاما و بدون فاصله وارد کنید (مثال: <span style="direction:ltr;display:inline-block">BTC-USD,AAPL,ETH-USD</span>)
    </div>
    """, unsafe_allow_html=True)
    tickers_input = st.text_input("نماد دارایی‌ها")
    start = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
    end = st.date_input("تاریخ پایان", value=pd.to_datetime("today"))
    download_btn = st.button("دریافت داده")

if download_btn and tickers_input.strip():
    tickers = [t.strip() for t in tickers_input.strip().split(",") if t.strip()]
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, group_by='ticker', auto_adjust=True)
        if data.empty:
            st.error("داده‌ای دریافت نشد!")
        else:
            new_downloaded = []
            for t in tickers:
                df, err = get_price_dataframe_from_yf(data, t)
                if df is not None:
                    df['Date'] = pd.to_datetime(df['Date'])
                    new_downloaded.append((t, df))
                    st.success(f"داده {t} با موفقیت دانلود شد.")
                else:
                    st.error(f"{err}")
            st.session_state["downloaded_dfs"].extend(new_downloaded)
    except Exception as ex:
        st.error(f"خطا در دریافت داده: {ex}")

with st.sidebar.expander("تنظیمات کلی ⚙️"):
    period = st.selectbox("بازه تحلیل بازده", ['ماهانه', 'سه‌ماهه', 'هفتگی'])
    resample_rule = {'ماهانه': 'M', 'سه‌ماهه': 'Q', 'هفتگی': 'W'}[period]
    annual_factor = {'ماهانه': 12, 'سه‌ماهه': 4, 'هفتگی': 52}[period]
    user_rf = st.number_input("نرخ بدون ریسک سالانه (%)", value=3.0, key="user_rf")

    # Use a form to safely handle investment_amount update
    with st.form(key="settings_form"):
        investment_amount = st.number_input("💵 سرمایه کل (دلار)", value=float(st.session_state["investment_amount"]), key="investment_amount_input")
        submit_button = st.form_submit_button(label="ثبت")
    if submit_button:
        st.session_state["investment_amount"] = investment_amount

# ---------- Minimum and Maximum Weight Constraints ----------
min_weights = []
max_weights = []
asset_names = []
if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
    name_counter = Counter()
    for t, df in st.session_state["downloaded_dfs"] + st.session_state["uploaded_dfs"]:
        base_name = t
        name_counter[base_name] += 1
        name = base_name if name_counter[base_name] == 1 else f"{base_name} ({name_counter[base_name]})"
        asset_names.append(name)

    with st.sidebar.expander("محدودیت وزن دارایی‌ها 🔒"):
        cols = st.columns(2)
        for i, name in enumerate(asset_names):
            with cols[i%2]:
                min_w = st.number_input(f"حداقل وزن {name}", value=0.0, key=f"minw_{name}")
                max_w = st.number_input(f"حداکثر وزن {name}", value=1.0, key=f"maxw_{name}")
                min_weights.append(min_w)
                max_weights.append(max_w)
    min_weights = np.array(min_weights)
    max_weights = np.array(max_weights)

# ---------- Main Analysis with Tabs ----------
tabs = st.tabs(["📊 آمار دارایی‌ها", "⚙️ معاملات و استراتژی‌ها", "📈 شبیه‌سازی پرتفوی", "🔮 پیش‌بینی قیمت"])

with tabs[0]:
    if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
        # 1- Prepare Price DataFrame
        name_counter = Counter()
        df_list = []
        asset_names = []
        for t, df in st.session_state["downloaded_dfs"] + st.session_state["uploaded_dfs"]:
            base_name = t
            name_counter[base_name] += 1
            name = base_name if name_counter[base_name] == 1 else f"{base_name} ({name_counter[base_name]})"
            temp_df = df.copy()
            temp_df = temp_df.rename(columns={"Price": name})
            temp_df = temp_df.dropna(subset=[name])
            temp_df = temp_df.set_index("Date")
            asset_names.append(name)
            df_list.append(temp_df[[name]])
        prices_df = pd.concat(df_list, axis=1, join="inner")
        if not isinstance(prices_df.index, pd.DatetimeIndex):
            prices_df.index = pd.to_datetime(prices_df.index)
        resampled_prices = prices_df.resample(resample_rule).last().dropna()

        # 2- Asset Statistics
        st.markdown("### 📊 آمار کلیدی دارایی‌ها")
        asset_stats = {}
        for name in asset_names:
            s = resampled_prices[name]
            asset_stats[name] = calc_asset_stats(s, freq=resample_rule, risk_free=user_rf)
        stats_df = pd.DataFrame(asset_stats).T[[
            'sharpe', 'sortino', 'volatility_ann', 'total_return_ann', 'implied_vol',
            'mean_ann', 'mean_month', 'std_ann', 'std_month', 'min_ann', 'min_month',
            'max_ann', 'max_month', 'var_95', 'cvar_95'
        ]]
        st.write(stats_df)
    else:
        st.warning("⚠️ لطفاً داده‌ای بارگذاری یا دریافت کنید.")

with tabs[1]:
    if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
        # 3- Options and Hedging Configuration
        st.markdown("### ⚙️ معاملات و استراتژی‌ها")
        option_rows_dict = {}
        for name in asset_names:
            with st.expander(f"📋 تنظیمات {name}"):
                opt_rows = []
                strategy = st.selectbox("استراتژی انتخابی", [
                    '-', 'Married Put', 'Protective Put', 'Covered Call', 'Collar',
                    'Bear Put Spread', 'Synthetic Put', 'Long Straddle/Strangle'
                ], key=f"strategy_{name}")
                current_price = resampled_prices[name].iloc[-1]
                purchase_price = st.number_input(f"قیمت خرید دارایی پایه ({name})", value=current_price, key=f"purchase_price_{name}")
                qty_asset = st.number_input(f"مقدار دارایی پایه ({name})", value=1.0, key=f"qty_asset_{name}")
                if strategy != '-':
                    if strategy in ['Married Put', 'Protective Put']:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            strike_put = st.number_input(f"قیمت اعمال پوت", value=current_price * 0.9, key=f"strike_put_{name}")
                        with col2:
                            premium_put = st.number_input(f"پریمیوم پوت (به ازای هر قرارداد)", value=0.0, key=f"premium_put_{name}")
                        with col3:
                            qty_contract = st.number_input(f"مقدار قرارداد پوت", value=1.0, key=f"qty_contract_put_{name}")
                        opt_rows.append(('خرید دارایی', 0, 0, qty_asset))
                        opt_rows.append(('خرید پوت', strike_put, premium_put, qty_contract))
                    elif strategy == 'Covered Call':
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            strike_call = st.number_input(f"قیمت اعمال کال", value=current_price * 1.1, key=f"strike_call_{name}")
                        with col2:
                            premium_call = st.number_input(f"پریمیوم کال (به ازای هر قرارداد)", value=0.0, key=f"premium_call_{name}")
                        with col3:
                            qty_contract = st.number_input(f"مقدار قرارداد کال", value=1.0, key=f"qty_contract_call_{name}")
                        opt_rows.append(('فروش کال', strike_call, premium_call, qty_contract))
                    elif strategy == 'Collar':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            strike_put = st.number_input(f"قیمت اعمال پوت", value=current_price * 0.9, key=f"strike_put_{name}")
                        with col2:
                            premium_put = st.number_input(f"پریمیوم پوت (به ازای هر قرارداد)", value=0.0, key=f"premium_put_{name}")
                        with col3:
                            strike_call = st.number_input(f"قیمت اعمال کال", value=current_price * 1.1, key=f"strike_call_{name}")
                        with col4:
                            premium_call = st.number_input(f"پریمیوم کال (به ازای هر قرارداد)", value=0.0, key=f"premium_call_{name}")
                        qty_contract = st.number_input(f"مقدار قرارداد پوت/کال", value=1.0, key=f"qty_contract_collar_{name}")
                        opt_rows.append(('خرید پوت', strike_put, premium_put, qty_contract))
                        opt_rows.append(('فروش کال', strike_call, premium_call, qty_contract))
                    elif strategy == 'Bear Put Spread':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            strike_put_high = st.number_input(f"قیمت اعمال پوت بالا", value=current_price, key=f"strike_put_high_{name}")
                        with col2:
                            premium_put_high = st.number_input(f"پریمیوم پوت بالا (به ازای هر قرارداد)", value=0.0, key=f"premium_put_high_{name}")
                        with col3:
                            strike_put_low = st.number_input(f"قیمت اعمال پوت پایین", value=current_price * 0.9, key=f"strike_put_low_{name}")
                        with col4:
                            premium_put_low = st.number_input(f"پریمیوم پوت پایین (به ازای هر قرارداد)", value=0.0, key=f"premium_put_low_{name}")
                        qty_contract = st.number_input(f"مقدار قرارداد پوت", value=1.0, key=f"qty_contract_bear_{name}")
                        opt_rows.append(('خرید پوت', strike_put_high, premium_put_high, qty_contract))
                        opt_rows.append(('فروش پوت', strike_put_low, premium_put_low, qty_contract))
                    elif strategy == 'Synthetic Put':
                        col1, col2 = st.columns(2)
                        with col1:
                            strike_call = st.number_input(f"قیمت اعمال کال", value=current_price, key=f"strike_call_{name}")
                        with col2:
                            premium_call = st.number_input(f"پریمیوم کال (به ازای هر قرارداد)", value=0.0, key=f"premium_call_{name}")
                        qty_contract = st.number_input(f"مقدار قرارداد کال", value=1.0, key=f"qty_contract_synth_{name}")
                        opt_rows.append(('فروش فیوچرز', 0, 0, qty_asset))
                        opt_rows.append(('خرید کال', strike_call, premium_call, qty_contract))
                    elif strategy == 'Long Straddle/Strangle':
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            strike_call = st.number_input(f"قیمت اعمال کال", value=current_price, key=f"strike_call_{name}")
                        with col2:
                            premium_call = st.number_input(f"پریمیوم کال (به ازای هر قرارداد)", value=0.0, key=f"premium_call_{name}")
                        with col3:
                            strike_put = st.number_input(f"قیمت اعمال پوت", value=current_price, key=f"strike_put_{name}")
                        with col4:
                            premium_put = st.number_input(f"پریمیوم پوت (به ازای هر قرارداد)", value=0.0, key=f"premium_put_{name}")
                        qty_contract = st.number_input(f"مقدار قرارداد کال/پوت", value=1.0, key=f"qty_contract_straddle_{name}")
                        opt_rows.append(('خرید کال', strike_call, premium_call, qty_contract))
                        opt_rows.append(('خرید پوت', strike_put, premium_put, qty_contract))
                option_rows_dict[name] = opt_rows
        st.session_state["option_rows"] = option_rows_dict.copy()

        # نمایش Payoff Diagram و تحلیل
        st.markdown("### 📊 نمودار پرداخت (Payoff Diagram)")
        for name in asset_names:
            if option_rows_dict.get(name):
                current_price = resampled_prices[name].iloc[-1]
                purchase_price = st.session_state[f"purchase_price_{name}"]
                qty_asset = st.session_state[f"qty_asset_{name}"]
                investment = purchase_price * qty_asset
                price_range = np.linspace(current_price * 0.5, current_price * 1.5, 100)
                payoffs = calculate_payoff(option_rows_dict[name], current_price, purchase_price, price_range)
                breakeven = calculate_breakeven(option_rows_dict[name], purchase_price)
                profit_loss_percent = calculate_profit_loss_percent(payoffs, purchase_price, investment)

                fig_payoff = go.Figure()
                fig_payoff.add_trace(go.Scatter(
                    x=price_range,
                    y=payoffs,
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='green' if payoffs[0] >= 0 else 'red'),
                    name=f'Payoff {name}',
                    hovertemplate='قیمت: %{x:.2f}<br>سود/زیان: %{y:.2f}<br>درصد: %{text:.2f}%<extra></extra>',
                    text=profit_loss_percent
                ))
                for i in range(len(payoffs) - 1):
                    if payoffs[i] * payoffs[i + 1] < 0:
                        fig_payoff.add_trace(go.Scatter(
                            x=[price_range[i], price_range[i + 1]],
                            y=[payoffs[i], payoffs[i + 1]],
                            mode='lines',
                            fill='tonexty',
                            line=dict(color='green' if payoffs[i + 1] >= 0 else 'red'),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                fig_payoff.add_trace(go.Scatter(
                    x=[breakeven, breakeven],
                    y=[min(payoffs), max(payoffs)],
                    mode='lines',
                    line=dict(color='blue', dash='dash'),
                    name='نقطه سربه‌سر'
                ))
                fig_payoff.add_trace(go.Scatter(
                    x=[current_price, current_price],
                    y=[min(payoffs), max(payoffs)],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='قیمت فعلی'
                ))
                fig_payoff.update_layout(
                    title=f"نمودار پرداخت برای {name}",
                    xaxis_title="قیمت دارایی",
                    yaxis_title="سود و زیان (دلار)",
                    template="plotly_white",
                    showlegend=True
                )
                st.plotly_chart(fig_payoff, use_container_width=True)

                st.markdown(f"### 📋 تحلیل استراتژی برای {name}")
                st.markdown(f"**نقطه سربه‌سر**: {breakeven:.2f} دلار")
                max_profit = max(payoffs)
                max_loss = min(payoffs)
                st.markdown(f"**حداکثر سود**: {format_money(max_profit)} ({format_percent(max_profit / investment)})")
                st.markdown(f"**حداکثر زیان**: {format_money(max_loss)} ({format_percent(max_loss / investment)})")

        if st.button("🔄 به‌روزرسانی"):
            st.rerun()

with tabs[2]:
    if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
        # 4- Calculate Returns with Options/Hedging
        returns_dict = {}
        for name in asset_names:
            price = resampled_prices[name]
            opt_rows = option_rows_dict.get(name, [])
            if opt_rows:
                ret_option = calc_options_series(opt_rows, price)
                returns_dict[name] = ret_option
            else:
                returns_dict[name] = price.pct_change().fillna(0)
        returns_df = pd.DataFrame(returns_dict).dropna()

        # 5- Portfolio Simulations
        st.markdown("### 📈 شبیه‌سازی پرتفوی")
        simulation_methods = {
            'Monte Carlo': {'metric': 'sharpe', 'color': 'Viridis', 'opt_crit': lambda x: np.argmax(x['sharpe'])},
            'VaR': {'metric': 'var_95', 'color': 'Plasma', 'opt_crit': lambda x: np.argmin(x['var_95'])},
            'CVaR': {'metric': 'cvar_95', 'color': 'Inferno', 'opt_crit': lambda x: np.argmin(x['cvar_95'])},
            'MPT': {'metric': 'sharpe', 'color': 'Viridis', 'opt_crit': lambda x: np.argmax(x['sharpe'])},
            'Equal Weight': {'metric': 'sharpe', 'color': 'Blues', 'opt_crit': lambda x: 0}
        }

        mean_returns = returns_df.mean() * annual_factor
        cov_matrix = returns_df.cov() * annual_factor
        n_portfolios = 2500
        for method, config in simulation_methods.items():
            all_risks, all_returns, all_weights, all_metrics = [], [], [], []
            cvar_alpha = 0.95
            if method == 'Equal Weight':
                weights = np.ones(len(asset_names)) / len(asset_names)
                if np.all(weights >= min_weights) and np.all(weights <= max_weights):
                    port_return = np.dot(weights, mean_returns)
                    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                    port_series = returns_df @ weights
                    sharpe = (port_return - user_rf) / port_std if port_std > 0 else 0
                    var_95 = var(port_series, cvar_alpha)
                    cvar_95 = cvar(port_series, cvar_alpha)
                    all_risks.append(port_std)
                    all_returns.append(port_return)
                    all_weights.append(weights)
                    all_metrics.append(sharpe if config['metric'] == 'sharpe' else -var_95 if config['metric'] == 'var_95' else -cvar_95)
            else:
                for i in range(n_portfolios):
                    valid = False
                    for _ in range(100):
                        ws = np.random.dirichlet(np.ones(len(asset_names)), size=1)[0]
                        if np.all(ws >= min_weights) and np.all(ws <= max_weights):
                            valid = True
                            break
                    if not valid:
                        continue
                    port_return = np.dot(ws, mean_returns)
                    port_std = np.sqrt(np.dot(ws.T, np.dot(cov_matrix, ws)))
                    port_series = returns_df @ ws
                    sharpe = (port_return - user_rf) / port_std if port_std > 0 else 0
                    var_95 = var(port_series, cvar_alpha)
                    cvar_95 = cvar(port_series, cvar_alpha)
                    all_risks.append(port_std)
                    all_returns.append(port_return)
                    all_weights.append(ws)
                    all_metrics.append(sharpe if config['metric'] == 'sharpe' else -var_95 if config['metric'] == 'var_95' else -cvar_95)

            all_risks = np.array(all_risks)
            all_returns = np.array(all_returns)
            all_weights = np.array(all_weights)
            all_metrics = np.array(all_metrics)

            # Efficient Frontier Plot
            st.markdown(f"#### مرز کارا - {method}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=all_risks*100, y=all_returns*100,
                mode='markers',
                marker=dict(
                    color=all_metrics, colorscale=config['color'], colorbar=dict(title=config['metric'].upper()),
                    size=7, line=dict(width=0)
                ),
                name='پرتفوی‌ها',
                hovertemplate=f'ریسک: %{{x:.2f}}٪<br>بازده: %{{y:.2f}}٪<br>{config["metric"].upper()}: %{{marker.color:.2f}}<extra></extra>'
            ))
            if method == 'MPT':
                max_sharpe_idx = config['opt_crit']({'sharpe': all_metrics})
                sharpe_star = all_metrics[max_sharpe_idx]
                max_risk = all_risks.max() * 1.3 * 100
                cal_x = np.linspace(0, max_risk, 100)
                cal_y = user_rf*100 + sharpe_star * cal_x
                fig.add_trace(go.Scatter(
                    x=cal_x, y=cal_y, mode='lines',
                    line=dict(dash='dash', color='red'), name='خط بازار سرمایه (CAL)'
                ))
            opt_idx = config['opt_crit']({'sharpe': all_metrics, 'var_95': all_metrics, 'cvar_95': all_metrics})
            fig.add_trace(go.Scatter(
                x=[all_risks[opt_idx]*100], y=[all_returns[opt_idx]*100],
                mode='markers+text', marker=dict(size=14, color='red'),
                text=[f"بهینه {method}"], textposition="top right", name=f"پرتفوی بهینه {method}"
            ))
            fig.update_layout(
                title=f"مرز کارا - {method}",
                xaxis_title="ریسک (%)",
                yaxis_title="بازده (%)",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(fig, use_container_width=True)

            # Pie Chart for Optimal Portfolio
            st.markdown(f"#### تخصیص دارایی - {method}")
            weights = all_weights[opt_idx]
            dollar_vals = weights * st.session_state["investment_amount"]
            cols = st.columns(len(asset_names))
            for i, name in enumerate(asset_names):
                percent = weights[i]
                dollar = dollar_vals[i]
                with cols[i]:
                    st.markdown(f"""
                    <div style='text-align:center;direction:rtl'>
                    <b>{name}</b><br>
                    {format_percent(percent)}<br>
                    {format_money(dollar)}
                    </div>
                    """, unsafe_allow_html=True)
            figpie = px.pie(
                values=dollar_vals,
                names=asset_names,
                title=f"توزیع دلاری پرتفوی بهینه - {method}",
                hole=0.4
            )
            st.plotly_chart(figpie, use_container_width=True)

with tabs[3]:
    if st.session_state["downloaded_dfs"] or st.session_state["uploaded_dfs"]:
        # 7- Price Forecasting
        st.markdown("### 🔮 پیش‌بینی قیمت دارایی‌ها")
        forecast_horizons = [1, 3, 6]
        for name in asset_names:
            st.markdown(f"#### پیش‌بینی برای {name}")
            forecasts = forecast_prices(resampled_prices[name], horizons=forecast_horizons)
            fig = go.Figure()
            last_price = resampled_prices[name].iloc[-1]
            fig.add_trace(go.Scatter(x=[0], y=[last_price], mode='markers', name='قیمت فعلی', marker=dict(size=10, color='blue')))
            for horizon in forecast_horizons:
                f = forecasts[horizon]
                fig.add_trace(go.Scatter(x=[horizon], y=[f['price']], mode='markers+text',
                                         text=[f"{f['price']:.2f} $"], textposition="top center",
                                         name=f'{horizon} ماه', marker=dict(size=8)))
                fig.add_trace(go.Scatter(x=[horizon, horizon], y=f['conf_int'], mode='lines',
                                         line=dict(color='gray', dash='dash'), name=f'بازه اطمینان {horizon} ماه'))
            fig.update_layout(
                title=f"پیش‌بینی قیمت {name}",
                xaxis_title="افق زمانی (ماه)",
                yaxis_title="قیمت پیش‌بینی‌شده ($)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
