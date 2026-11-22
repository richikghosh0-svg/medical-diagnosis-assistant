import cv2
import numpy as np
import tensorflow as tf

_model = None

def _load():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model("models/skin_model.keras")
    return _model

def predict_skin(image_bgr, threshold=0.5):
    """image_bgr: an image as read by cv2.
    The model has its own preprocessing built in, so give it raw 0-255 pixels."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype("float32")
    p_malignant = float(_load().predict(img[None], verbose=0)[0][0])
    return {
        "Malignant": p_malignant,
        "Benign": 1 - p_malignant,
        "Label": "Malignant" if p_malignant >= threshold else "Benign",
    }