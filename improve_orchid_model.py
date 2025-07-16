import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import re

# === Confusion Matrix and Classification Report ===
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Paths ===
base_dir = 'D:/CP/ML_Model/orchid-dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# === Parameters ===
img_size = (224, 224)
batch_size = 32
additional_epochs = 10

# Base model name without version suffix (just the prefix)
base_model_name = 'orchid_disease_model'

# Print debug info for where it looks
print("Looking for models with base name:", base_model_name)
print("Current directory contents:", os.listdir('.'))

# === Data Generators ===
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=30, zoom_range=0.3,
                               width_shift_range=0.2, height_shift_range=0.2,
                               shear_range=0.2, horizontal_flip=True, fill_mode='nearest')
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
test_data = test_gen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

# === Compute Class Weights ===
labels = train_data.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# === Find latest model version ===
def get_latest_version(base_name, directory='.'):
    existing = [f for f in os.listdir(directory) if f.startswith(base_name)]
    versions = []
    for name in existing:
        match = re.search(r'_v(\d+)$', name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions, default=0)

latest_version = get_latest_version(base_model_name)
print("Existing versions found:", [f for f in os.listdir('.') if f.startswith(base_model_name)])
print("Latest version:", latest_version)

if latest_version == 0:
    print(f"No base model found! Make sure '{base_model_name}_v1' exists first.")
    exit()

prev_model_path = os.path.join(os.path.dirname(__file__), f"{base_model_name}_v{latest_version}")
print(f"Loading model from: {prev_model_path}")
model = load_model(prev_model_path)
print(f"Loaded model from: {prev_model_path}")

# === Unfreeze some layers of base model ===
base_model = model.layers[0]
base_model.trainable = True
fine_tune_at = 200
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
print("Unfroze top layers for fine-tuning")

# === Recompile with fine-tune settings ===
model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# === Add callbacks ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
]

# === Continue Training ===
history = model.fit(train_data, epochs=additional_epochs, validation_data=val_data, callbacks=callbacks, class_weight=class_weights_dict)

# === Evaluate ===
loss, acc = model.evaluate(test_data)
print("Test accuracy after training: {:.2f}".format(acc))

# === Save as new version ===
new_version = latest_version + 1
new_model_name = f"{base_model_name}_v{new_version}"
model.save(new_model_name)
print(f"Saved model as: {new_model_name}")

# === Export to TFLite ===
converter = tf.lite.TFLiteConverter.from_saved_model(new_model_name)
tflite_model = converter.convert()

tflite_path = f"{new_model_name}.tflite"
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model exported as: {tflite_path}")


# === Summary of Training ===



# Predict class probabilities and labels
y_pred_probs = model.predict(test_data)
y_pred = y_pred_probs.argmax(axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
