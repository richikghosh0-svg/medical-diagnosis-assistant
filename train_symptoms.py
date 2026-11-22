"""Train the symptom -> disease model (tabular data, so Random Forest, not a CNN).

Use your own data: dataset/symptoms.csv with one 0/1 column per symptom and a
'disease' column. If the file is missing, a small SYNTHETIC demo dataset is generated
so the pipeline runs end-to-end. Demo data is for testing the code only.

Run:  python train_symptoms.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

CSV = Path("dataset/symptoms.csv")
MODEL_PATH = Path("models/symptom_model.joblib")

# Rough symptom profiles, used ONLY to build the synthetic demo dataset.
DEMO_PROFILES = {
    "Influenza": ["fever", "cough", "fatigue", "body_ache", "headache", "sore_throat"],
    "Common Cold": ["cough", "sore_throat", "runny_nose", "sneezing"],
    "COVID-19": ["fever", "cough", "fatigue", "loss_of_smell", "shortness_of_breath", "headache"],
    "Migraine": ["headache", "nausea", "light_sensitivity", "fatigue"],
    "Gastroenteritis": ["nausea", "vomiting", "diarrhea", "fever", "stomach_pain"],
    "Allergic Rhinitis": ["sneezing", "runny_nose", "itchy_eyes", "cough"],
}


def make_demo_data(n_per_class=300, seed=0):
    rng = np.random.default_rng(seed)
    symptoms = sorted({s for v in DEMO_PROFILES.values() for s in v})
    rows = []
    for disease, profile in DEMO_PROFILES.items():
        for _ in range(n_per_class):
            row = {s: 0 for s in symptoms}
            for s in profile:
                row[s] = int(rng.random() < 0.8)           # typical symptoms
            for s in symptoms:
                if s not in profile:
                    row[s] = int(rng.random() < 0.05)      # noise
            row["disease"] = disease
            rows.append(row)
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed)
    CSV.parent.mkdir(exist_ok=True)
    df.to_csv(CSV, index=False)
    print("Generated synthetic demo dataset at", CSV)
    return df


def main():
    df = pd.read_csv(CSV) if CSV.exists() else make_demo_data()
    X, y = df.drop(columns=["disease"]), df["disease"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    print(classification_report(y_te, clf.predict(X_te)))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "features": list(X.columns)}, MODEL_PATH)
    print("Saved", MODEL_PATH)


if __name__ == "__main__":
    main()
