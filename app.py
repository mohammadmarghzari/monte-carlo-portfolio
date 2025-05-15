import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# تنظیمات صفحه
st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو (سالانه)", layout="wide")
st.title("📈 ابزار تحلیل پرتفو با روش مونت‌کارلو (سالانه)")
st.markdown("ریسک هر دارایی = ۲۰٪ سالانه | هدف: ساخت پرتفو با ریسک سالانه نزدیک به ۳۰٪")

# تابع خواندن فایل CSV
def read_csv_file(file):
    try:
        df = pd.read_csv(
            file,
            encoding='utf-8',
            sep=',',
            decimal='.',
            thousands=None,
            na_values=['', 'NA', 'N/A', 'null', '-', 'NaN'],
            skipinitialspace=True,
            on_bad_lines='warn'
        )
        return df
    except UnicodeDecodeError:
        st.error(f"خطا در رمزگذاری فایل {file.name}. لطفاً فایل را با رمزگذاری UTF-8 ذخیره کنید.")
        return None
    except Exception as e:
        st.error(f"خطا در خواندن فایل {file.name}: {e}")
        return None

# تابع یافتن ستون تاریخ
def find_date_column(df, file_name):
    possible_cols = [
        col for col in df.columns 
        if any(key in col.lower() for key in ['date', 'time', 'timestamp'])
    ]
    if possible_cols:
        return possible_cols[0]
    st.warning(f"ستونی مشابه 'Date' در فایل {file_name} یافت نشد.")
    return None

# تابع یافتن ستون قیمت
def find_price_column(df, file_name):
    possible_cols = [
        col for col in df.columns 
        if any(key in col.lower() for key in ['price', 'close', 'adj close', 'adjusted close'])
    ]
    if possible_cols:
        return possible_cols[0]
    st.warning(f"ستونی مشابه 'Price' یا 'Close' در فایل {file_name} یافت نشد.")
    return None

# سایدبار برای آپلود فایل‌ها
st.sidebar.header("📂 بارگذاری فایل دارایی‌ها (CSV)")
uploaded_files = st.sidebar.file_uploader(
    "چند فایل CSV آپلود کنید (هر دارایی یک فایل)",
    type=['csv'],
    accept_multiple_files=True
)

if uploaded_files:
    prices_df = pd.DataFrame()
    asset_names = []
    date_column = None

    # پردازش فایل‌های آپلودشده
    for file in uploaded_files:
        df = read_csv_file(file)
        if df is None:
            continue
            
        name = file.name.split('.')[0]
        st.write(f"📄 فایل: {name} - ستون‌ها: {list(df.columns)}")

        # یافتن ستون تاریخ
        date_col = find_date_column(df, name)
        if not date_col:
            st.write("لطفاً ستون تاریخ را به‌صورت دستی انتخاب کنید:")
            date_col = st.selectbox(
                f"انتخاب ستون تاریخ برای {name}",
                options=df.columns,
                key=f"date_col_{name}"
            )
        
        # تبدیل تاریخ
        try:
            # فرمت تاریخ نمونه: '03/08/2025' (MM/DD/YYYY)
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='%m/%d/%Y')
            invalid_dates = df[df[date_col].isna()]
            if not invalid_dates.empty:
                st.warning(f"⚠️ {len(invalid_dates)} مقدار تاریخ نامعتبر در فایل '{name}' یافت شد:")
                st.write("نمونه تاریخ‌های نامعتبر:")
                st.dataframe(invalid_dates[[date_col]].head())
                df = df.dropna(subset=[date_col])
                st.info(f"ردیف‌های مشکل‌دار حذف شدند. {len(df)} ردیف باقی ماند.")
            if df.empty:
                st.error(f"❌ هیچ تاریخ معتبری در فایل '{name}' باقی نماند.")
                continue
        except Exception as e:
            st.error(f"❌ خطا در تبدیل ستون '{date_col}' به تاریخ: {e}")
            continue

        # یافتن یا انتخاب ستون قیمت
        close_col = find_price_column(df, name)
        if not close_col:
            st.write("لطفاً ستون قیمت را به‌صورت دستی انتخاب کنید:")
            close_col = st.selectbox(
                f"انتخاب ستون قیمت برای {name}",
                options=df.columns,
                key=f"price_col_{name}"
            )
        
        # پاکسازی داده‌های قیمت
        df[close_col] = df[close_col].astype(str).str.replace(',', '', regex=False)
        df[close_col] = pd.to_numeric(df[close_col], errors='coerce')
        invalid_prices = df[df[close_col].isna()]
        if not invalid_prices.empty:
            st.warning(f"⚠️ {len(invalid_prices)} مقدار نامعتبر در ستون '{close_col}' فایل '{name}' یافت شد:")
            st.write("نمونه مقادیر نامعتبر:")
            st.dataframe(invalid_prices[[date_col, close_col]].head())
            df = df.dropna(subset=[close_col])
            st.info(f"ردیف‌های مشکل‌دار حذف شدند. {len(df)} ردیف باقی ماند.")
        if df.empty:
            st.error(f"❌ هیچ قیمت معتبری در فایل '{name}' باقی نماند.")
            continue
        
        # تنظیم تاریخ به‌عنوان شاخص و ادغام داده‌ها
        df = df[[date_col, close_col]].set_index(date_col)
        df.columns = [name]
        if prices_df.empty:
            prices_df = df
            date_column = date_col
        else:
            prices_df = prices_df.join(df, how='inner')
            if prices_df.empty:
                st.error(f"❌ هیچ تاریخ مشترکی بین فایل‌ها یافت نشد. لطفاً تاریخ‌ها را بررسی کنید.")
                st.stop()

        asset_names.append(name)
        st.success(f"✅ ستون‌های انتخاب‌شده برای {name}: تاریخ='{date_col}'، قیمت='{close_col}'")

    if prices_df.empty or len(asset_names) < 1:
        st.error("❌ هیچ داده معتبری از فایل‌ها استخراج نشد. لطفاً فایل‌ها و ستون‌ها را بررسی کنید.")
        st.stop()

    # محاسبه بازده و کوواریانس سالانه
    prices_df.index = pd.to_datetime(prices_df.index)
    returns = prices_df.pct_change().dropna()  # بازده روزانه
    mean_returns = returns.mean() * 252  # بازده مورد انتظار سالانه
    cov_matrix = returns.cov() * 252  # کوواریانس سالانه

    # شبیه‌سازی مونت‌کارلو
    np.random.seed(42)
    n_portfolios = 10000
    n_assets = len(asset_names)
    results = np.zeros((3 + n_assets, n_portfolios))

    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        port_return = np.dot(weights, mean_returns)  # بازده سالانه
        port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # ریسک سالانه
        sharpe_ratio = port_return / port_std  # نسبت شارپ سالانه

        results[0, i] = port_return
        results[1, i] = port_std
        results[2, i] = sharpe_ratio
        results[3:, i] = weights

    # انتخاب پرتفو با ریسک سالانه نزدیک به ۳۰٪
    target_risk = 0.30  # ریسک هدف سالانه
    best_idx = np.argmin(np.abs(results[1] - target_risk))

    best_return = results[0, best_idx]
    best_risk = results[1, best_idx]
    best_sharpe = results[2, best_idx]
    best_weights = results[3:, best_idx]

    # نمایش نتایج
    st.subheader("📊 نتایج پرتفو پیشنهادی (سالانه)")
    st.markdown(f"""
    - ✅ **بازده مورد انتظار سالانه:** {best_return:.2%}  
    - ⚠️ **ریسک سالانه:** {best_risk:.2%}  
    - 🧠 **نسبت شارپ:** {best_sharpe:.2f}  
    """)

    for i, name in enumerate(asset_names):
        st.markdown(f"🔹 **وزن {name}:** {best_weights[i]*100:.2f}٪")

    # نمودار پراکندگی پرتفوها (تعاملی)
    st.subheader("📈 نمودار پراکندگی پرتفوها (بازده در مقابل ریسک)")
    portfolio_df = pd.DataFrame({
        'ریسک سالانه (%)': results[1] * 100,
        'بازده سالانه (%)': results[0] * 100,
        'نسبت شارپ': results[2]
    })
    fig = px.scatter(
        portfolio_df,
        x='ریسک سالانه (%)',
        y='بازده سالانه (%)',
        color='نسبت شارپ',
        color_continuous_scale='Viridis',
        opacity=0.6,
        title='پراکندگی پرتفوهای شبیه‌سازی‌شده',
        hover_data={'ریسک سالانه (%)': ':.2f', 'بازده سالانه (%)': ':.2f', 'نسبت شارپ': ':.2f'}
    )
    fig.add_scatter(
        x=[best_risk * 100],
        y=[best_return * 100],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        name='پرتفوی بهینه'
    )
    fig.update_layout(
        xaxis_title="ریسک سالانه (%)",
        yaxis_title="بازده مورد انتظار سالانه (%)",
        showlegend=True,
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)

    # نمودار سود/زیان (تعاملی)
    st.subheader("📈 نمودار سود/زیان پرتفو نسبت به تغییر قیمت‌ها")
    price_changes = np.linspace(-0.5, 0.5, 100)
    total_change = np.zeros_like(price_changes)
    for i, w in enumerate(best_weights):
        total_change += w * price_changes

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_changes * 100,
            y=total_change * 100,
            mode='lines',
            name='تغییر ارزش پرتفو',
            line=dict(color='blue')
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.update_layout(
        title="نمودار سود/زیان پرتفو",
        xaxis_title="درصد تغییر قیمت دارایی‌ها",
        yaxis_title="درصد سود/زیان پرتفو",
        showlegend=True,
        hovermode='x'
    )
    st.plotly_chart(fig, use_container_width=True)

    # نمودار وزنی پرتفو (تعاملی - دوناتی)
    st.subheader("📊 نمودار وزنی پرتفو")
    weights_df = pd.DataFrame({
        'دارایی': asset_names,
        'وزن (%)': best_weights * 100
    })
    fig = px.pie(
        weights_df,
        names='دارایی',
        values='وزن (%)',
        hole=0.4,
        title='وزن دارایی‌ها در پرتفوی بهینه'
    )
    fig.update_traces(textinfo='percent+label', hoverinfo='label+percent')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("لطفاً چند فایل CSV قیمت دارایی‌ها آپلود کنید.")
