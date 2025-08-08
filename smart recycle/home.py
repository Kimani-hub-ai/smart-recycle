# edge_ai_recyclable_classifier.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Step 1: Data Preprocessing
# Use ImageDataGenerator for basic augmentation and rescaling
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training and validation datasets
train_dir = 'recyclable_dataset/'  # should contain subfolders like 'plastic', 'paper', etc.

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Step 2: Model Architecture
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

# Step 3: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 4: Train the Model
epochs = 10
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data
)

# Step 5: Evaluate Accuracy
val_loss, val_accuracy = model.evaluate(val_data)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")

# Step 6: Save and Convert the Model to TFLite
model.save('recyclable_classifier.h5')

# Convert to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('recyclable_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model successfully converted to TensorFlow Lite!")
