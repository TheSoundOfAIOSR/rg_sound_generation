import os
import tensorflow as tf
from .tfrecord_provider import CompleteTFRecordProvider
from .compute_measures import get_measures
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


def map_features(features):
    conf = LocalConfig()

    # name = features["sample_name"]
    note_number = features["note_number"]
    velocity = features["velocity"]
    instrument_id = features["instrument_id"]

    h_freq = features["h_freq"]
    h_mag = features["h_mag"]
    h_phase = features["h_phase"]

    h_freq = tf.io.parse_tensor(h_freq, out_type=tf.string)
    h_mag = tf.io.parse_tensor(h_mag, out_type=tf.string)
    h_phase = tf.io.parse_tensor(h_phase, out_type=tf.string)

    h_freq = tf.io.parse_tensor(h_freq, out_type=tf.float32)
    h_mag = tf.io.parse_tensor(h_mag, out_type=tf.float32)
    h_phase = tf.io.parse_tensor(h_phase, out_type=tf.float32)

    h_freq = tf.expand_dims(h_freq, axis=0)
    h_mag = tf.expand_dims(h_mag, axis=0)
    h_phase = tf.expand_dims(h_phase, axis=0)

    normalized_data, mask = \
        conf.data_handler.normalize(h_freq, h_mag, h_phase, note_number)

    for k, v in normalized_data.items():
        normalized_data[k] = tf.squeeze(v, axis=0)

    mask = tf.squeeze(mask, axis=0)

    measures = get_measures(h_freq, h_mag, conf)

    if conf.use_one_hot_conditioning:
        note_number = tf.one_hot(
            note_number - conf.starting_midi_pitch, depth=conf.num_pitches)
        velocity = tf.cast(velocity, dtype=tf.float32) / 25.0 - 1.0
        velocity = tf.one_hot(
            tf.cast(velocity, dtype=tf.uint8), depth=conf.num_velocities)
        instrument_id = tf.one_hot(
            instrument_id, depth=conf.num_instruments)
    else:
        note_number = tf.cast(note_number, dtype=tf.float32) / 127.0
        velocity = tf.cast(velocity, dtype=tf.float32) / 127.0
        num_instruments = float(conf.num_instruments)
        instrument_id = tf.cast(instrument_id, tf.float32) / num_instruments

    data = {
        "mask": mask,
        "note_number": tf.squeeze(note_number),
        "velocity": tf.squeeze(velocity),
        "instrument_id": tf.squeeze(instrument_id),
        "measures": measures,
    }
    data.update(normalized_data)
    return data


def get_dataset(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    train_path = os.path.join(conf.dataset_dir, "train.tfrecord")
    valid_path = os.path.join(conf.dataset_dir, "valid.tfrecord")
    test_path = os.path.join(conf.dataset_dir, "test.tfrecord")

    # train_dataset = create_dataset(train_path, map_func=None, batch_size=1)
    #
    # iterator = iter(train_dataset)
    # for i in range(5):
    #     d = next(iterator)
    #     for k, v in d.items():
    #         d[k] = tf.squeeze(v, axis=0)
    #     ne = map_features(d)

    train_dataset = create_dataset(train_path, map_func=map_features, batch_size=conf.batch_size)
    valid_dataset = create_dataset(valid_path, map_func=map_features, batch_size=conf.batch_size)
    test_dataset = create_dataset(test_path, map_func=map_features, batch_size=conf.batch_size)

    return train_dataset, valid_dataset, test_dataset
