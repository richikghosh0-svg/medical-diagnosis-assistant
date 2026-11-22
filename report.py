"""PDF report generation (fpdf2). Text is kept ASCII so the built-in font works."""
from datetime import datetime

from fpdf import FPDF

DISCLAIMER = ("This report is produced by a machine-learning model for educational "
              "purposes only. It is not a medical diagnosis. Please consult a qualified "
              "healthcare professional.")


def _ascii(text):
    return str(text).encode("ascii", "replace").decode("ascii")


def build_report(patient_name="Anonymous", symptoms=None, symptom_preds=None, xray_result=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Patient Analysis Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now():%Y-%m-%d %H:%M}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Patient: {_ascii(patient_name)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    if symptoms:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Symptom analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, "Reported symptoms: " + _ascii(", ".join(symptoms)))
        for disease, p in (symptom_preds or []):
            pdf.cell(0, 6, f"  - Possible condition: {_ascii(disease)} ({p:.0%} model confidence)",
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    if xray_result:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Chest X-ray analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 6, f"Model finding: {_ascii(xray_result['label'])} "
                       f"({xray_result['confidence']:.0%} confidence)",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "Recommendation", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Consult a healthcare professional for evaluation. "
                         "Seek urgent care for breathing difficulty, chest pain, or confusion.")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 5, DISCLAIMER)
    return bytes(pdf.output())
