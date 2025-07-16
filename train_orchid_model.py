import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths
base_dir = 'D:/CP/ML_Model/orchid-dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Parameters
img_size = (224, 224)
batch_size = 32
epochs = 10

# Data generators with augmentation on train only
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical'
)
test_data = test_gen.flow_from_directory(
    test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
)

# Model setup
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(img_size[0], img_size[1], 3)
)
base_model.trainable = False  # Freeze base model initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data
)

# Evaluate on test set
loss, acc = model.evaluate(test_data)
print(f"Test accuracy: {acc:.2f}")

# Save the TensorFlow SavedModel
model.save('orchid_disease_model')

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model('orchid_disease_model')
tflite_model = converter.convert()

# Save the TFLite model file
with open('orchid_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ TFLite model exported as orchid_disease_model.tflite")
