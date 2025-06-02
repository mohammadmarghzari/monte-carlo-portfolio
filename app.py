import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

st.set_page_config(page_title="تحلیل پرتفو با مونت‌کارلو", layout="wide")
st.title("تحلیل پرتفو بیت‌کوین و اتریوم با مونت‌کارلو")

def smart_read_file(file):
    content = file.read()
    try:
        content = content.decode("utf-8")
    except:
        content = content.decode("latin1")
    seps = [',',';','|','\t']
    sep_counts = [(s, content.count(s)) for s in seps]
    sep = max(sep_counts, key=lambda x:x[1])[0] if max(sep_counts, key=lambda x:x[1])[1] > 0 else ','
    lines = [l.strip() for l in content.splitlines() if l.strip() and not l.strip().startswith('#')]
    if not lines:
        return None
    header = [x.strip().replace('"','').replace("'",'') for x in re.split(sep, lines[0])]
    def find_col(colnames, candidates):
        for c in candidates:
            for i, h in enumerate(colnames):
                if c.lower() in h.lower():
                    return i
        return None
    date_idx = find_col(header, ['date', 'تاریخ'])
    price_idx = find_col(header, ['price', 'قیمت'])
    if date_idx is None or price_idx is None:
        return None
    data_rows = []
    for row in lines[1:]:
        parts = [x.strip().replace('"','').replace("'",'') for x in re.split(sep, row)]
        if len(parts) <= max(date_idx, price_idx): continue
        date_val = parts[date_idx]
        price_val = parts[price_idx]
        price_val = price_val.replace(' ', '').replace(',', '')
        price_val = re.sub(r'[^\d\.\-]', '', price_val)
        try:
            price_float = float(price_val)
        except:
            continue
        data_rows.append([date_val, price_float])
    if not data_rows:
        return None
    df = pd.DataFrame(data_rows, columns=['Date', 'Price'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'Price'])
    df = df.sort_values('Date')
    if len(df) < 3:
        return None
    return df

uploaded_files = st.file_uploader("دو فایل CSV دیتا را آپلود کن (BTC و ETH)", type=['csv'], accept_multiple_files=True)

if uploaded_files and len(uploaded_files) >= 2:
    dfs = []
    names = []
    for file in uploaded_files:
        df = smart_read_file(file)
        if df is not None:
            name = file.name.replace(".csv","")
            dfs.append(df)
            names.append(name)
            st.write(f"پیش‌نمایش داده {name}:")
            st.dataframe(df.tail(10))
        else:
            st.error(f"خواندن فایل {file.name} با مشکل روبرو شد.")
    if len(dfs) == 2:
        df_total = dfs[0].set_index("Date").join(dfs[1].set_index("Date"), lsuffix='_1', rsuffix='_2', how='inner')
        st.write("جدول نهایی (تاریخ مشترک):", df_total.shape)
        st.dataframe(df_total.tail(10))
        if len(df_total) < 10:
            st.error("تعداد سطرهای مشترک خیلی کم است! داده‌های هر دو دارایی باید بازه زمانی مشترک کافی داشته باشند.")
        else:
            st.success(f"جدول نهایی {len(df_total)} سطر دارد و آماده تحلیل است.")
            # محاسبه بازده
            returns = df_total.pct_change().dropna()
            st.write("پیش‌نمایش بازده‌ها:")
            st.dataframe(returns.tail())
            # ماتریس کوواریانس
            cov_matrix = returns.cov()
            st.write("ماتریس کوواریانس:")
            st.dataframe(cov_matrix)
            mean_returns = returns.mean()
            n_portfolios = 4000
            results = np.zeros((3 + len(names), n_portfolios))
            for i in range(n_portfolios):
                weights = np.random.random(2)
                weights /= np.sum(weights)
                port_return = np.dot(weights, mean_returns)
                port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = port_return / port_std if port_std != 0 else 0
                results[0, i] = port_return
                results[1, i] = port_std
                results[2, i] = sharpe
                results[3, i] = weights[0]
                results[4, i] = weights[1]
            results_df = pd.DataFrame(results.T, columns=['بازده','ریسک','شارپ',f'وزن_{names[0]}',f'وزن_{names[1]}'])
            st.write("نمونه پرتفوهای شبیه‌سازی‌شده:")
            st.dataframe(results_df.head())
            # پیدا کردن بهترین پرتفوها
            max_sharpe = results_df.iloc[results_df['شارپ'].idxmax()]
            min_risk = results_df.iloc[results_df['ریسک'].idxmin()]
            st.markdown("### پرتفو با بیشترین شارپ:")
            st.write(max_sharpe)
            st.markdown("### پرتفو با کمترین ریسک:")
            st.write(min_risk)
            st.markdown("### نمودار ریسک/بازده پرتفوها")
            fig = px.scatter(results_df, x='ریسک', y='بازده', color='شارپ', title='پرتفوهای مونت‌کارلو')
            fig.add_scatter(x=[max_sharpe['ریسک']], y=[max_sharpe['بازده']], mode='markers', marker=dict(color='red', size=15, symbol='star'), name="بیشترین شارپ")
            fig.add_scatter(x=[min_risk['ریسک']], y=[min_risk['بازده']], mode='markers', marker=dict(color='blue', size=12, symbol='circle'), name="کمترین ریسک")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### نمودار دایره‌ای وزن دارایی‌های پرتفو بهینه (بیشترین شارپ)")
            st.plotly_chart(px.pie(names=[names[0],names[1]], values=[max_sharpe[f'وزن_{names[0]}'], max_sharpe[f'وزن_{names[1]}']], title="وزن دارایی‌ها"), use_container_width=True)
else:
    st.info("حداقل دو فایل معتبر (BTC و ETH) آپلود کن.")
