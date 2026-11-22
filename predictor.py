"""Inference helpers used by the Streamlit app."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

MODEL_DIR = Path("models")
IMG_SIZE = (224, 224)


@lru_cache(maxsize=1)
def _xray_model():
    from tensorflow import keras
    model = keras.models.load_model(MODEL_DIR / "xray_model.keras")
    names = json.loads((MODEL_DIR / "class_names.json").read_text())
    return model, names


@lru_cache(maxsize=1)
def _symptom_bundle():
    return joblib.load(MODEL_DIR / "symptom_model.joblib")


def predict_xray(image):
    """image: PIL.Image or file path. Returns {'label', 'confidence', 'probabilities'}."""
    model, names = _xray_model()
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    arr = np.asarray(image.convert("RGB").resize(IMG_SIZE), dtype="float32")
    probs = model.predict(arr[None, ...], verbose=0)[0]   # raw 0-255: scaling is inside the model
    top = int(np.argmax(probs))
    return {
        "label": names[top],
        "confidence": float(probs[top]),
        "probabilities": {n: float(p) for n, p in zip(names, probs)},
    }


def symptom_list():
    return _symptom_bundle()["features"]


def predict_symptoms(selected, top_k=3):
    """selected: list of symptom names. Returns top_k [(disease, probability), ...]."""
    bundle = _symptom_bundle()
    vec = np.array([[1 if f in selected else 0 for f in bundle["features"]]])
    probs = bundle["model"].predict_proba(vec)[0]
    order = np.argsort(probs)[::-1][:top_k]
    classes = bundle["model"].classes_
    return [(str(classes[i]), float(probs[i])) for i in order]
