import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Layer
from itertools import groupby
import csv

# Create data pipeline
training_data = tf.keras.utils.image_dataset_from_directory('data_spectrogram', batch_size=32, image_size=(515, 71),
                                                            shuffle='false', labels='inferred', label_mode='categorical', seed=42, validation_split=.2, subset='training')
validation_data = tf.keras.utils.image_dataset_from_directory('data_spectrogram', batch_size=32, image_size=(515, 71),
                                                              shuffle='false', labels='inferred', label_mode='categorical', seed=42, validation_split=.2, subset='validation')

# Build the model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(515, 71, 3)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='Adam', loss='CategoricalCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.summary()

# Create weights for classes
num_launchers = len(os.listdir(r'data_spectrogram/launchers'))
num_planes = len(os.listdir(r'data_spectrogram/planes'))
num_rifles = len(os.listdir(r'data_spectrogram/rifles'))
num_soldiers = len(os.listdir(r'data_spectrogram/soldiers'))
num_vehicles = len(os.listdir(r'data_spectrogram/vehicles'))
num_total = (num_launchers + num_planes + num_rifles + num_soldiers + num_vehicles)

weight_for_launchers = (1 / num_launchers) * (num_total / 2.0)
weight_for_planes = (1 / num_planes) * (num_total / 2.0)
weight_for_rifles = (1 / num_rifles) * (num_total / 2.0)
weight_for_soldiers = (1 / num_soldiers) * (num_total / 2.0)
weight_for_vehicles = (1 / num_vehicles) * (num_total / 2.0)

class_weight = {0: weight_for_launchers, 1: weight_for_planes, 2: weight_for_rifles, 3: weight_for_soldiers, 4: weight_for_vehicles}

# Fit the model
hist = model.fit(training_data, epochs=16, validation_data=validation_data, class_weight=class_weight)

plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()
