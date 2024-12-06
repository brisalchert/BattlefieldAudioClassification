import os
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf  # Need version 2.10.0 for GPU support on Windows
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

# Create data pipeline
training_data, validation_data = tf.keras.utils.image_dataset_from_directory(
    'data_spectrogram',
    batch_size=64,
    image_size=(515, 71),
    shuffle=True,
    labels='inferred',
    label_mode='categorical',
    seed=42,
    validation_split=.2,
    subset='both'
)

# Build the model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(515, 71, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='CategoricalCrossentropy',
    metrics=['accuracy']
)
model.summary()

# Create weights for classes
y_train = np.concatenate([y for x, y in training_data], axis=0)

num_launchers = np.sum([y.argmax() == 0 for y in y_train])
num_planes = np.sum([y.argmax() == 1 for y in y_train])
num_rifles = np.sum([y.argmax() == 2 for y in y_train])
num_soldiers = np.sum([y.argmax() == 3 for y in y_train])
num_vehicles = np.sum([y.argmax() == 4 for y in y_train])
num_total = (num_launchers + num_planes + num_rifles + num_soldiers + num_vehicles)

weight_for_launchers = (1 / num_launchers) * (num_total / 2.0)
weight_for_planes = (1 / num_planes) * (num_total / 2.0)
weight_for_rifles = (1 / num_rifles) * (num_total / 2.0)
weight_for_soldiers = (1 / num_soldiers) * (num_total / 2.0)
weight_for_vehicles = (1 / num_vehicles) * (num_total / 2.0)

class_weight = {
    0: weight_for_launchers,
    1: weight_for_planes,
    2: weight_for_rifles,
    3: weight_for_soldiers,
    4: weight_for_vehicles
}

# Fit the model
hist = model.fit(training_data, epochs=20, validation_data=validation_data, class_weight=class_weight)

plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.yscale('log')
plt.show()

plt.title('Accuracy')
plt.plot(hist.history['accuracy'], 'r')
plt.plot(hist.history['val_accuracy'], 'b')
plt.show()
