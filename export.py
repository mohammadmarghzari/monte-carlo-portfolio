import pandas as pd
from io import BytesIO

def export_excel(results_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        results_df.to_excel(writer, sheet_name="Summary")
    output.seek(0)
    return output

def export_pdf(results_df):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="گزارش خلاصه پرتفو", ln=True, align="C")
    for idx, row in results_df.iterrows():
        pdf.cell(200, 10, txt=f"{row['نام دارایی']}: {row['وزن (%)']:.2f}%", ln=True)
    output = BytesIO()
    pdf.output(output)
    output.seek(0)
    return output
