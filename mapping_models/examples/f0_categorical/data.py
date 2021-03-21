import os
import tensorflow as tf
import numpy as np

from tfrecord_provider import CompleteTFRecordProvider


with open('pitch_classes.txt', 'r') as f:
    data = f.read().splitlines()

pitch_labels = data[:12]
pitch_values = [float(x) for x in data[12:]]


def complete_record_generator(dataset_dir, set_name, batch_size):
    tfrecord_path = os.path.join(dataset_dir, set_name, 'complete.tfrecord*')
    return iter(CompleteTFRecordProvider(tfrecord_path).get_batch(batch_size=batch_size))


def data_generator(complete_data_generator):
    while True:
        features = next(complete_data_generator)
        velocity = (features['velocity'] - 25) / 25
        velocity_categorical = tf.keras.utils.to_categorical(velocity, num_classes=5)
        instrument_source = tf.keras.utils.to_categorical(features['instrument_source'], num_classes=3)
        qualities = features['qualities']
        f0_hz = features['f0_hz']
        f0_categorical = tf.keras.utils.to_categorical(np.digitize(f0_hz, bins=pitch_values),
                                                       num_classes=len(pitch_values))
        batch_size = features['z'].shape[0]

        inputs = {
            'velocity': velocity_categorical,
            'instrument_source': instrument_source,
            'qualities': qualities,
            'z': tf.reshape(features['z'], (batch_size, 1000, 16))
        }

        outputs = {
            'f0_categorical': f0_categorical
        }
        yield inputs, outputs
