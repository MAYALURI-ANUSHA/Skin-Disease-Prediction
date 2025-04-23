import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Dataset path
data_dir = r"C:\Users\sai ganesh\.cache\kagglehub\datasets\ismailpromus\skin-diseases-image-dataset\versions\1\image-classification-project\data"

# Updated batch size
batch_size = 32

# Load full dataset
train_data_full = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224, 224),
    batch_size=batch_size
)

val_data_full = image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=batch_size
)

# Limit to small subset for faster training
train_data = train_data_full.take(4)  # 4 batches × 32 = 128 images
val_data = val_data_full.take(2)      # 2 batches × 32 = 64 images

# Get number of classes
num_classes = len(train_data_full.class_names)

# Load base model
base_model = tf.keras.applications.EfficientNetB7(
    include_top=False, input_shape=(224, 224, 3), weights="imagenet"
)
base_model.trainable = True

# Freeze early layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Data augmentation
data_aug = tf.keras.Sequential([
    layers.RandomWidth(0.2),
    layers.RandomHeight(0.2),
    layers.RandomRotation(0.2),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
], name="data_augmentation")

# Model architecture
inputs = layers.Input(shape=(224, 224, 3))
x = data_aug(inputs, training=True)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(1024, activation="relu", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(512, activation="relu", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(256, activation="relu", kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(num_classes)(x)
outputs = layers.Activation("softmax", dtype=tf.float32)(x)

model = Model(inputs, outputs)

# Compile model
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]
)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, min_delta=0.00001)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4, min_lr=1e-8)

# Train model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[early_stop, reduce_lr]
)

# Save model
model.save(r"C:\Users\sai ganesh\.cache\kagglehub\datasets\ismailpromus\skin-diseases-image-dataset\versions\1\image-classification-project\models\trained_model.keras")
