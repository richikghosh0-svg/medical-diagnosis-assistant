"""Challenge 2: Skin lesion (benign vs malignant) classifier with transfer learning.

Expected folders:
dataset/skin/train/benign      dataset/skin/test/benign
dataset/skin/train/malignant   dataset/skin/test/malignant
"""
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

os.makedirs("models", exist_ok=True)

IMG = (224, 224)
BATCH = 32
DATA = "dataset/skin"

train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA}/train", image_size=IMG, batch_size=BATCH, label_mode="binary",
    validation_split=0.2, subset="training", seed=42)
val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA}/train", image_size=IMG, batch_size=BATCH, label_mode="binary",
    validation_split=0.2, subset="validation", seed=42)
test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DATA}/test", image_size=IMG, batch_size=BATCH, label_mode="binary",
    shuffle=False)

print("Class order:", train_ds.class_names)  # ['benign', 'malignant'] -> malignant = 1

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
])

# Frozen ImageNet backbone + small trainable head (fast on CPU)
base = MobileNetV2(input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=IMG + (3,))
x = augment(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # scales to [-1, 1]
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

metrics = [
    "accuracy",
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc"),
]

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("models/skin_model.keras", save_best_only=True),
]

print("Stage 1: training the new head")
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy", metrics=metrics)
model.fit(train_ds, validation_data=val_ds, epochs=8, callbacks=callbacks)

print("Stage 2: fine-tuning the top of the backbone")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="binary_crossentropy", metrics=metrics)
model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

print("Test set evaluation:")
results = model.evaluate(test_ds)
print(dict(zip(model.metrics_names, results)))

model.save("models/skin_model.keras")
print("Saved models/skin_model.keras")