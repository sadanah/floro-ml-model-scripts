import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load data
train_ds = tf.keras.utils.image_dataset_from_directory(
    'orchid-dataset/train',
    label_mode="categorical",
    image_size=(224, 224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    'orchid-dataset/valid',
    label_mode="categorical",
    image_size=(224, 224),
    batch_size=32
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),

    Dropout(0.25),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Add early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,  # Let early stopping decide when to stop
    callbacks=[early_stop]
)

# Save the model
model.save("orchid_custom_cnn.keras")

# Evaluate on validation set
val_loss, val_acc = model.evaluate(val_ds)
print("Validation Accuracy:", val_acc)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("orchid_custom_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved as orchid_custom_model.tflite")

# --- PLOTS & METRICS ---

# Accuracy plot
plt.plot(history.history['accuracy'], label='Train Acc', color='blue')
plt.plot(history.history['val_accuracy'], label='Val Acc', color='orange')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# Confusion matrix + classification report
# Reload validation set without shuffle
val_ds_noshuffle = tf.keras.utils.image_dataset_from_directory(
    'orchid-dataset/valid',
    label_mode="categorical",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Get class names
class_names = val_ds_noshuffle.class_names

# Get true and predicted labels
true_labels = tf.concat([y for x, y in val_ds_noshuffle], axis=0)
y_true = tf.argmax(true_labels, axis=1)

y_pred_probs = model.predict(val_ds_noshuffle)
y_pred = tf.argmax(y_pred_probs, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
