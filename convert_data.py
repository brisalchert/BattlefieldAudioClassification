import os
import shutil
import tensorflow as tf
import soundfile

# Remove old data directory
shutil.rmtree('data')

# Create data directory
if not os.path.exists('data'):
    os.makedirs(r'data\vehicles')
    os.makedirs(r'data\launchers')
    os.makedirs(r'data\planes')
    os.makedirs(r'data\rifles')
    os.makedirs(r'data\soldiers')

# Convert sound files to 16-bit PCMs
TRAIN_ORIGINAL = os.path.join('data_original', 'clean_train')
TEST_ORIGINAL = os.path.join('data_original', 'clean_test')
DATA_CONVERTED = os.path.dirname('data')

vehicles = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\Military_vehicles*.wav', TEST_ORIGINAL+r'\Military_vehicles*.wav'))
launchers = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\Missile_launchers*.wav', TEST_ORIGINAL+r'\Missile_launchers*.wav'))
planes = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\plane*.wav', TEST_ORIGINAL+r'\plane*.wav'))
rifles = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\rifles*.wav', TEST_ORIGINAL+r'\rifles*.wav'))
soldiers = tf.data.Dataset.list_files((TRAIN_ORIGINAL+r'\soldiers*.wav', TEST_ORIGINAL+r'\soldiers*.wav'))


def convert_files(filelist, dirname):
    for file in filelist:
        file = file.numpy()
        file_name = os.path.basename(file)
        data, samplerate = soundfile.read(file)
        soundfile.write(fr'data\{dirname}\{file_name.decode()}', data, samplerate, subtype='PCM_16')


convert_files(vehicles, 'vehicles')
convert_files(launchers, 'launchers')
convert_files(planes, 'planes')
convert_files(rifles, 'rifles')
convert_files(soldiers, 'soldiers')
