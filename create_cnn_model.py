import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
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

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
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

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

