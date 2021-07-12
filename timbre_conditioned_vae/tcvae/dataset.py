import os
import tensorflow as tf
from .tfrecord_provider import CompleteTFRecordProvider
from .localconfig import LocalConfig


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

    note_number = features["note_number"]
    instrument_id = tf.one_hot(features["instrument_id"], depth=conf.num_instruments)
    velocity = tf.cast(features["velocity"], dtype=tf.float32) / 25. - 1.
    velocity = tf.one_hot(tf.cast(velocity, dtype=tf.uint8), depth=conf.num_velocities)

    h_freq = features["h_freq"]
    h_mag = features["h_mag"]

    h_freq = tf.io.parse_tensor(h_freq, out_type=tf.string)
    h_mag = tf.io.parse_tensor(h_mag, out_type=tf.string)

    h_freq = tf.io.parse_tensor(h_freq, out_type=tf.float32)
    h_mag = tf.io.parse_tensor(h_mag, out_type=tf.float32)

    harmonics = tf.shape(h_freq)[-1]

    h_freq = tf.expand_dims(h_freq, axis=0)
    h_mag = tf.expand_dims(h_mag, axis=0)

    f0_shifts, mag_env, h_freq_shifts, h_mag_distribution, mask = \
        conf.data_handler.normalize(h_freq, h_mag, note_number)

    h_mag_orig = tf.expand_dims(pad_function(tf.squeeze(h_mag), conf), axis=-1)
    h_freq_orig = tf.expand_dims(pad_function(tf.squeeze(h_freq), conf), axis=-1)

    f0_shifts = tf.squeeze(f0_shifts, axis=0)
    mag_env = tf.squeeze(mag_env, axis=0)
    h_freq_shifts = tf.squeeze(h_freq_shifts, axis=0)
    h_mag_distribution = tf.squeeze(h_mag_distribution, axis=0)
    mask = tf.squeeze(mask, axis=0)

    freq = tf.concat([f0_shifts, h_freq_shifts], axis=-1)
    mag = tf.concat([mag_env, h_mag_distribution], axis=-1)

    freq = tf.expand_dims(pad_function(freq, conf), axis=-1)
    mag = tf.expand_dims(pad_function(mag, conf), axis=-1)
    mask = pad_function(mask, conf)

    h = tf.concat([freq, mag], axis=-1)

    note_number = tf.one_hot(note_number - conf.starting_midi_pitch, depth=conf.num_pitches)

    return {
        "instrument_id": tf.squeeze(instrument_id),
        "velocity": tf.squeeze(velocity),
        "note_number": tf.squeeze(note_number),
        "h": h,
        "mask": mask,
        "h_mag_orig": h_mag_orig,
        "h_freq_orig": h_freq_orig,
        "harmonics": harmonics
    }


def get_dataset(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    train_path = os.path.join(conf.dataset_dir, "train.tfrecord")
    valid_path = os.path.join(conf.dataset_dir, "valid.tfrecord")
    test_path = os.path.join(conf.dataset_dir, "test.tfrecord")

    train_dataset = create_dataset(train_path, map_func=map_features, batch_size=conf.batch_size)
    valid_dataset = create_dataset(valid_path, map_func=map_features, batch_size=conf.batch_size)
    test_dataset = create_dataset(test_path, map_func=map_features, batch_size=conf.batch_size)

    return train_dataset, valid_dataset, test_dataset
