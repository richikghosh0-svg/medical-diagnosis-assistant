# Medical Diagnosis Assistant

Educational decision-support demo: chest X-ray classification (CNN, transfer learning),
symptom-based disease prediction (Random Forest), PDF reports, Streamlit UI.
**Not a medical device and not a substitute for professional advice.**

## Setup
```bash
pip install -r requirements.txt
```
1. Download "Chest X-Ray Images (Pneumonia)" from Kaggle and place it at
   `dataset/chest_xray/{train,test}/{NORMAL,PNEUMONIA}`.
2. (Optional) put your own `dataset/symptoms.csv` (0/1 symptom columns + `disease`).
   Without it, `train_symptoms.py` builds a synthetic demo set.

## Run
```bash
python train.py            # X-ray model  -> models/xray_model.keras
python train_symptoms.py   # symptom model -> models/symptom_model.joblib
streamlit run app.py
```

## Design notes
- MobileNetV2 transfer learning (frozen head, then fine-tune) instead of a 1-layer CNN.
- Pixel normalisation is a layer inside the model, so training and inference can't diverge.
- Class weights handle the Normal/Pneumonia imbalance; evaluation reports recall,
  precision, F1 and a confusion matrix, not just accuracy.
- No horizontal-flip augmentation (chest anatomy is not symmetric).
