import pandas as pd
from io import BytesIO

def format_money(val):
    if val == 0:
        return "۰ دلار"
    elif abs(val) >= 1:
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

def download_excel(results, asset_names):
    df = pd.DataFrame(results)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Summary')
    output.seek(0)
    return output

def validate_investment_input(val):
    try:
        val = float(str(val).replace('٫', '.').replace(',', ''))
        if val >= 0:
            return True, val
        else:
            return False, 0
    except Exception:
        return False, 0
