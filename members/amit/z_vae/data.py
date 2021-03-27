import os
import numpy as np

from tfrecord_provider import CompleteTFRecordProvider


def data_generator(dataset_dir, set_name, batch_size):
    tfrecord_path = os.path.join(dataset_dir, set_name, 'complete.tfrecord*')
    data_iterable = iter(CompleteTFRecordProvider(tfrecord_path).get_batch(batch_size=batch_size))

    while True:
        features = next(data_iterable)
        z = np.reshape(features['z'], (batch_size, 1000, 16))
        z_pad = np.zeros((batch_size, 24, 16))
        z_concat = (np.concatenate([z, z_pad], axis=1) + 7.) / 14.
        yield z_concat, z_concat
