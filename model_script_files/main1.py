import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt 


# Paths to dataset folders
base_dir = '/c:/Users/sai ganesh/Downloads/skin diseases.v3i.folder/dataset'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

# Image dimensions and batch size
img_height, img_width = 150, 150
batch_size = 32
num_classes = 3  # Change as per your dataset

# Helper to load images and labels
def load_data_from_folder(base_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(base_path))
    class_indices = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    for cls in class_names:
        cls_path = os.path.join(base_path, cls)
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            img = Image.open(img_path).resize((img_width, img_height)).convert('RGB')
            images.append(np.array(img))
            labels.append(class_indices[cls])
    
    return np.array(images), to_categorical(np.array(labels), num_classes=num_classes)

# Load datasets
x_train, y_train = load_data_from_folder(r'C:\Users\sai ganesh\Downloads\skin diseases.v3i.folder\train')
x_val, y_val = load_data_from_folder(r'C:\Users\sai ganesh\Downloads\skin diseases.v3i.folder\train')
x_test, y_test = load_data_from_folder(r'C:\Users\sai ganesh\Downloads\skin diseases.v3i.folder\train')

# Normalize pixel values
x_train = x_train / 255.0
x_val = x_val / 255.0
x_test = x_test / 255.0

# Create data generators
train_datagen = ImageDataGenerator()
valid_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
valid_generator = valid_datagen.flow(x_val, y_val, batch_size=batch_size)
test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=epochs
)

# Save the model
model_path = '/c:/Users/sai ganesh/Downloads/skin diseases.v3i.folder/model.h5'
model.save(model_path)
print(f"Model saved to {model_path}")

# Load and test the model
loaded_model = load_model(model_path)
test_loss, test_acc = loaded_model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")

# Predict on test data
predictions = loaded_model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Map predicted labels to class names
class_indices = test_generator.class_indices
class_names = {v: k for k, v in class_indices.items()}
predicted_labels = [class_names[i] for i in predicted_classes]

# Print predictions
print("Predictions:")
for i, label in enumerate(predicted_labels):
    print(f"Image {i}: {label}")