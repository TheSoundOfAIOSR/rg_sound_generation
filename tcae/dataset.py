import tensorflow as tf
import os
from tcae.localconfig import LocalConfig
from dataset.read_dataset import TFRecordProvider


def create_dataset(
        dataset_path,
        batch_size=16,
        audio_length=64000,
        map_func=None):
    assert os.path.isfile(dataset_path)

    train_data_provider = TFRecordProvider(
        file_pattern=dataset_path,
        audio_length=audio_length,
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

    name = features["sample_name"]
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

    max_harmonics = conf.data_handler.max_harmonics
    harmonics = tf.shape(h_freq)[-1]

    normalized_data = conf.data_handler.normalize(
        h_freq, h_mag, h_phase, note_number)

    h_freq, h_mag, _ = conf.data_handler.denormalize(
        normalized_data, phase_mode='none')
    measures = conf.data_handler.compute_measures(h_freq, h_mag)

    # zero-padding to max_harmonics
    h_freq = tf.pad(
        h_freq,
        paddings=((0, 0), (0, 0), (0, max_harmonics - harmonics)))

    h_mag = tf.pad(
        h_mag,
        paddings=((0, 0), (0, 0), (0, max_harmonics - harmonics)))

    h_phase = tf.pad(
        h_phase,
        paddings=((0, 0), (0, 0), (0, max_harmonics - harmonics)))

    h_freq = tf.squeeze(h_freq, axis=0)
    h_mag = tf.squeeze(h_mag, axis=0)
    h_phase = tf.squeeze(h_phase, axis=0)

    for k, v in normalized_data.items():
        normalized_data[k] = tf.squeeze(v, axis=0)

    measures = dict((k, tf.squeeze(v, axis=0)) for k, v in measures.items())

    if conf.use_one_hot_conditioning:
        note_number = tf.one_hot(
            note_number - conf.starting_midi_pitch, depth=conf.num_pitches)
        velocity = tf.cast(velocity, dtype=tf.float32) / 25.0 - 1.0
        velocity = tf.one_hot(
            tf.cast(velocity, dtype=tf.uint8), depth=conf.num_velocities)
        instrument_id = tf.one_hot(
            instrument_id, depth=conf.num_instruments)
    else:
        num_pitches = float(conf.num_pitches)
        num_velocities = float(conf.num_velocities)
        num_instruments = float(conf.num_instruments)
        starting_midi_pitch = float(conf.starting_midi_pitch)

        note_number = tf.cast(note_number, dtype=tf.float32)
        velocity = tf.cast(velocity, dtype=tf.float32)
        instrument_id = tf.cast(instrument_id, dtype=tf.float32)

        note_number = note_number - starting_midi_pitch
        velocity = tf.math.round(velocity / 25.0) - 1.0

        note_number = note_number / num_pitches
        velocity = velocity / num_velocities
        instrument_id = instrument_id / num_instruments

    inputs = normalized_data.copy()
    inputs.update({
        "name": name,
        "note_number": tf.squeeze(note_number),
        "velocity": tf.squeeze(velocity),
        "instrument_id": tf.squeeze(instrument_id),
        "measures": [measures[k] for k in conf.data_handler.measure_names],
    })

    targets = normalized_data.copy()
    targets.update({
        "h_freq": h_freq,
        "h_mag": h_mag,
        "h_phase": h_phase,
    })
    targets.update(measures)

    return inputs, targets


def get_dataset(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    train_path = os.path.join(conf.dataset_dir, "train.tfrecord")
    valid_path = os.path.join(conf.dataset_dir, "valid.tfrecord")
    # test_path = os.path.join(conf.dataset_dir, "test.tfrecord")

    # train_dataset = create_dataset(train_path, map_func=None, batch_size=1)
    #
    # iterator = iter(train_dataset)
    # for i in range(5):
    #     d = next(iterator)
    #     for k, v in d.items():
    #         d[k] = tf.squeeze(v, axis=0)
    #     ne = map_features(d)

    train_dataset = create_dataset(
        train_path, map_func=map_features, batch_size=conf.batch_size)
    valid_dataset = create_dataset(
        valid_path, map_func=map_features, batch_size=conf.batch_size)
    # test_dataset = create_dataset(
    #     test_path, map_func=map_features, batch_size=conf.batch_size)

    if conf.dataset_modifier is not None:
        train_dataset, valid_dataset, test_dataset = conf.dataset_modifier(
            train_dataset, valid_dataset, valid_dataset)

    return train_dataset, valid_dataset, valid_dataset
