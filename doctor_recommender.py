"""Challenge 5: doctor recommendation engine.
1) map condition -> specialty  2) filter doctors  3) rank by weighted score.
The doctors in data/doctors.csv are SAMPLE data only."""
import pandas as pd

CONDITION_TO_SPECIALTY = {
    "pneumonia": "Pulmonologist", "covid": "Pulmonologist",
    "influenza": "General Physician", "flu": "General Physician",
    "cold": "General Physician", "dengue": "General Physician",
    "migraine": "Neurologist", "diabetes": "Endocrinologist",
    "hypertension": "Cardiologist", "blood pressure": "Cardiologist",
    "malignant": "Dermatologist", "benign": "Dermatologist",
    "skin": "Dermatologist", "mole": "Dermatologist",
}


def specialty_for(condition):
    c = condition.lower().strip()
    for key, specialty in CONDITION_TO_SPECIALTY.items():
        if key in c:
            return specialty
    return "General Physician"


def _norm(s):
    return (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else s * 0 + 1


def recommend(condition, city=None, max_fee=None, language=None, top_n=3,
              csv_path="data/doctors.csv"):
    df = pd.read_csv(csv_path)
    specialty = specialty_for(condition)
    df = df[(df["specialty"] == specialty) & (df["available"] == "Yes")]
    if city:
        df = df[df["city"].str.lower() == city.strip().lower()]
    if max_fee:
        df = df[df["fee"] <= max_fee]
    if language:
        df = df[df["languages"].str.contains(language, case=False)]
    if df.empty:
        return specialty, df
    df = df.copy()
    # weights: rating 50%, experience 30%, affordability 20%
    df["score"] = (0.5 * _norm(df["rating"]) + 0.3 * _norm(df["experience_years"])
                   + 0.2 * (1 - _norm(df["fee"])))
    return specialty, df.sort_values("score", ascending=False).head(top_n)