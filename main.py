import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from itertools import groupby
import csv

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000Hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# Create datasets for each type of sound
vehicles = tf.data.Dataset.list_files(r'data\Military_vehicles*.wav')  # r creates raw string literal
launchers = tf.data.Dataset.list_files(r'data\Missile_launchers*.wav')
planes = tf.data.Dataset.list_files(r'data\plane*.wav')
rifles = tf.data.Dataset.list_files(r'data\rifles*.wav')
soldiers = tf.data.Dataset.list_files(r'data\soldiers*.wav')

# assign values to each classification
class_names = ['vehicle', 'launcher', 'plane', 'rifle', 'soldier']

vehicles = tf.data.Dataset.zip((vehicles,
                                tf.data.Dataset.from_tensor_slices(tf.fill(len(vehicles), class_names.index('vehicle')))))
launchers = tf.data.Dataset.zip((launchers,
                                 tf.data.Dataset.from_tensor_slices(tf.fill(len(launchers), class_names.index('launcher')))))
planes = tf.data.Dataset.zip((planes,
                              tf.data.Dataset.from_tensor_slices(tf.fill(len(planes), class_names.index('plane')))))
rifles = tf.data.Dataset.zip((rifles,
                              tf.data.Dataset.from_tensor_slices(tf.fill(len(rifles), class_names.index('rifle')))))
soldiers = tf.data.Dataset.zip((soldiers,
                                tf.data.Dataset.from_tensor_slices(tf.fill(len(soldiers), class_names.index('soldier')))))

# Concatenate data
data = vehicles.concatenate(launchers.concatenate(planes.concatenate(rifles.concatenate(soldiers))))

# preprocessing function for spectrogram
def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:80000]  # Shorten to 5 seconds at 16000 Hz maximum
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)  # Pad with zeroes if shorter than 5 seconds
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)  # Add third dimension for CNN
    return spectrogram, label  # Spectrogram dimensions: (2491, 257, 1)

filepath, label = data.shuffle(buffer_size=5).as_numpy_iterator().next()
spectrogram, label = preprocess(filepath, label)
plt.figure(figsize=(30, 20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()

# Create data pipeline
data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)

# Split data into training and testing (80/20, total of 122 batches)
train = data.take(98)
test = data.skip(98).take(24)

# Build the model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(2491, 257, 1)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile('Adam', loss='CategoricalCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
model.summary()
