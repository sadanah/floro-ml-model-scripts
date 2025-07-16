import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load saved model
model = tf.keras.models.load_model("orchid_custom_cnn.keras")

# Load validation set (no shuffle!)
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

# Predict
y_pred_probs = model.predict(val_ds_noshuffle)
y_pred = tf.argmax(y_pred_probs, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
