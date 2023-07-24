import os
import shutil
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import soundfile


# Function for converting original files to 16-bit PCM format
def convert_files(filelist, dirname):
    for file in filelist:
        file = file.numpy()
        file_name = os.path.basename(file)
        data, samplerate = soundfile.read(file)
        soundfile.write(fr'data\{dirname}\{file_name.decode()}', data, samplerate, subtype='PCM_16')


# Preprocessing function for loading waveform
# Credit for referenced code: https://github.com/nicknochnack/DeepAudioClassification
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


# Preprocessing function for spectrogram
# Credit for referenced code: https://github.com/nicknochnack/DeepAudioClassification
def preprocess(file_path):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:80000]  # Shorten to 5 seconds at 16000 Hz maximum
    zero_padding = tf.zeros([80000] - tf.shape(wav), dtype=tf.float32)  # Pad with zeroes if shorter than 5 seconds
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)  # Add third dimension for CNN
    return spectrogram  # Spectrogram dimensions: (2491, 257, 1)


# Function for writing spectrogram images to their respective locations in the file directory
def generate_spectrograms(filelist, dirname):
    for file in filelist:
        file = file.numpy()
        file_name = os.path.splitext(os.path.basename(file))[0].decode()
        spectrogram = preprocess(file)
        plt.figure()
        plt.imshow(tf.transpose(tf.math.log(spectrogram))[0])
        plt.axis('off')
        plt.savefig(fr'data_spectrogram\{dirname}\{file_name}.jpg', bbox_inches='tight')
        plt.close()


# Remove old data directory
if os.path.exists('data'):
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

convert_files(vehicles, 'vehicles')
convert_files(launchers, 'launchers')
convert_files(planes, 'planes')
convert_files(rifles, 'rifles')
convert_files(soldiers, 'soldiers')

# Remove old spectrogram data directory
if os.path.exists('data_spectrogram'):
    shutil.rmtree('data_spectrogram')

# Create spectrogram data directory
if not os.path.exists('data_spectrogram'):
    os.makedirs(r'data_spectrogram\vehicles')
    os.makedirs(r'data_spectrogram\launchers')
    os.makedirs(r'data_spectrogram\planes')
    os.makedirs(r'data_spectrogram\rifles')
    os.makedirs(r'data_spectrogram\soldiers')

# Create file lists for each sound type
vehicles = tf.data.Dataset.list_files(r'data\vehicles\Military_vehicles*.wav')
launchers = tf.data.Dataset.list_files(r'data\launchers\Missile_launchers*.wav')
planes = tf.data.Dataset.list_files(r'data\planes\plane*.wav')
rifles = tf.data.Dataset.list_files(r'data\rifles\rifles*.wav')
soldiers = tf.data.Dataset.list_files(r'data\soldiers\soldiers*.wav')

# Create spectrograms
generate_spectrograms(vehicles, 'vehicles')
generate_spectrograms(launchers, 'launchers')
generate_spectrograms(planes, 'planes')
generate_spectrograms(rifles, 'rifles')
generate_spectrograms(soldiers, 'soldiers')
