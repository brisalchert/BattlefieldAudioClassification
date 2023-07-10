import os
import shutil
import tensorflow as tf
import soundfile

# Remove old data directory
shutil.rmtree('data')

# Create data directory
if not os.path.exists('data'):
    os.makedirs('data')

# Convert sound files to 16-bit PCMs
TRAIN_ORIGINAL = os.path.join('data_original', 'clean_train')
TEST_ORIGINAL = os.path.join('data_original', 'clean_test')
DATA_CONVERTED = os.path.dirname('data')

vehicles = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\Military_vehicles*.wav', TEST_ORIGINAL+r'\Military_vehicles*.wav'))
launchers = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\Missile_launchers*.wav', TEST_ORIGINAL+r'\Missile_launchers*.wav'))
planes = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\plane*.wav', TEST_ORIGINAL+r'\plane*.wav'))
rifles = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\rifles*.wav', TEST_ORIGINAL+r'\rifles*.wav'))
soldiers = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\soldiers*.wav', TEST_ORIGINAL+r'\soldiers*.wav'))

data = vehicles.concatenate(launchers.concatenate(planes.concatenate(rifles.concatenate(soldiers))))

for file in data:
    file = file.numpy()
    file_name = os.path.basename(file)
    data, samplerate = soundfile.read(file)
    soundfile.write(fr'data\{file_name.decode()}', data, samplerate, subtype='PCM_16')
