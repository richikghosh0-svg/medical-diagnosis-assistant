"""Train the chest X-ray classifier (Normal vs Pneumonia) with transfer learning.

Expected folder layout (Kaggle "Chest X-Ray Images (Pneumonia)"):
    dataset/chest_xray/train/{NORMAL,PNEUMONIA}
    dataset/chest_xray/test/{NORMAL,PNEUMONIA}

Run:  python train.py
"""
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = Path("dataset/chest_xray")
MODEL_DIR = Path("models")
IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42


def load_data():
    # 80/20 split of the train folder (Kaggle's own "val" folder has only 16 images).
    train_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH, label_mode="categorical")
    val_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH, label_mode="categorical")
    test_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR / "test", image_size=IMG_SIZE, batch_size=BATCH,
        label_mode="categorical", shuffle=False)
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE),
            test_ds.prefetch(AUTOTUNE), class_names)


def class_weights(class_names):
    """The dataset is imbalanced (~3x more pneumonia), so weight the rare class up."""
    counts = [len(list((DATA_DIR / "train" / c).glob("*"))) for c in class_names]
    total = sum(counts)
    return {i: total / (len(counts) * n) for i, n in enumerate(counts)}


def build_model():
    augment = keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="augmentation")  # no horizontal flip: anatomy is not left/right symmetric

    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)   # normalisation lives INSIDE the model
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    return keras.Model(inputs, outputs), base


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )


def evaluate(model, test_ds, class_names):
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds), axis=1)
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_true, y_pred))
    print("\n", classification_report(y_true, y_pred, target_names=class_names))


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    train_ds, val_ds, test_ds, class_names = load_data()
    weights = class_weights(class_names)
    print("Classes:", class_names, "| class weights:", weights)

    model, base = build_model()
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
        keras.callbacks.ModelCheckpoint(MODEL_DIR / "xray_model.keras", save_best_only=True),
    ]

    # Phase 1: train only the new classification head
    compile_model(model, 1e-3)
    model.fit(train_ds, validation_data=val_ds, epochs=1,
              class_weight=weights, callbacks=callbacks)

    # Phase 2: fine-tune the top of the pretrained network at a low learning rate
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    compile_model(model, 1e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=5,
              class_weight=weights, callbacks=callbacks)

    evaluate(model, test_ds, class_names)
    model.save(MODEL_DIR / "xray_model.keras")
    (MODEL_DIR / "class_names.json").write_text(json.dumps(class_names))
    print("Saved model to", MODEL_DIR)


if __name__ == "__main__":
    main()
