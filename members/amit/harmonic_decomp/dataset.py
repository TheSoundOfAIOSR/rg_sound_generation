import os
import tensorflow as tf
import numpy as np
from tfrecord_provider import CompleteTFRecordProvider
import tsms
from localconfig import LocalConfig


def create_dataset(
        dataset_path,
        batch_size=16,
        example_secs=4,
        sample_rate=16000,
        frame_rate=250,
        map_func=None
):
    assert os.path.isfile(dataset_path)

    train_data_provider = CompleteTFRecordProvider(
        file_pattern=dataset_path,
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        map_func=map_func
    )

    dataset = train_data_provider.get_batch(
        batch_size,
        shuffle=True,
        repeats=1
    )

    return dataset


def pad_function(sample, conf: LocalConfig):
    return tf.pad(sample,
                  tf.convert_to_tensor([[0, conf.row_dim - conf.harmonic_frame_steps],
                                        [0, conf.col_dim - tf.shape(sample)[1]]]))


def map_features(features):
    conf = LocalConfig()

    note_number = tf.one_hot(features["note_number"] - 40, depth=conf.num_pitches)
    instrument_id = tf.one_hot(features["instrument_id"], depth=conf.num_instruments)
    velocity = tf.cast(features["velocity"], dtype=tf.float32) / 25. - 1.
    velocity = tf.one_hot(tf.cast(velocity, dtype=tf.uint8), depth=conf.num_velocities)

    h_freq = features["h_freq"]
    h_mag = features["h_mag"]

    h_freq = tf.io.parse_tensor(h_freq, out_type=tf.string)
    h_mag = tf.io.parse_tensor(h_mag, out_type=tf.string)

    h_freq = tf.io.parse_tensor(h_freq, out_type=tf.float32)
    h_mag = tf.io.parse_tensor(h_mag, out_type=tf.float32)

    h_freq = tf.expand_dims(h_freq, axis=0)
    h_mag = tf.expand_dims(h_mag, axis=0)

    f0 = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)
    f0_mean = tf.math.reduce_mean(f0, axis=1)
    harmonics = tf.shape(h_freq)[-1]
    harmonic_indices = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_indices = harmonic_indices[tf.newaxis, tf.newaxis, :]
    h_freq_centered = h_freq - (f0_mean * harmonic_indices)

    h_mag = tf.squeeze(h_mag, axis=0)
    h_freq_centered = tf.squeeze(h_freq_centered, axis=0)

    h_freq_norm = h_freq_centered / f0_mean
    # (h_freq_centered - tf.reduce_mean(h_freq_centered)) / tf.math.reduce_std(h_freq_centered)
    h_mag_norm = h_mag / tf.math.reduce_max(h_mag)

    h_freq_norm = tf.expand_dims(pad_function(h_freq_norm, conf), axis=-1)
    h_mag_norm = tf.expand_dims(pad_function(h_mag_norm, conf), axis=-1)

    h = tf.concat([h_freq_norm, h_mag_norm], axis=-1)

    mask = tf.ones_like(h_freq_centered)
    mask = pad_function(mask, conf)

    return {
        "instrument_id": tf.squeeze(instrument_id),
        "velocity": tf.squeeze(velocity),
        "note_number": tf.squeeze(note_number),
        "h": h,
        "mask": mask
    }


def get_dataset(conf: LocalConfig):
    train_path = os.path.join(conf.dataset_dir, "train.tfrecord")
    valid_path = os.path.join(conf.dataset_dir, "valid.tfrecord")
    test_path = os.path.join(conf.dataset_dir, "test.tfrecord")

    train_dataset = create_dataset(train_path, map_func=map_features, batch_size=conf.batch_size)
    valid_dataset = create_dataset(valid_path, map_func=map_features, batch_size=conf.batch_size)
    test_dataset = create_dataset(test_path, map_func=map_features, batch_size=1)

    return train_dataset, valid_dataset, test_dataset
