import tensorflow as tf
import os
from tcae.localconfig import LocalConfig


class CompleteTFRecordProvider:
    def __init__(self,
                 file_pattern,
                 example_secs=4,
                 sample_rate=16000,
                 frame_rate=250,
                 map_func=None):
        self._file_pattern = file_pattern
        self._sample_rate = sample_rate
        self._frame_rate = frame_rate
        self._audio_length = example_secs * sample_rate
        self._feature_length = example_secs * frame_rate
        self._data_format_map_fn = tf.data.TFRecordDataset
        self._map_func = map_func

    def get_dataset(self, shuffle=True):
        def parse_tfexample(record):
            features = tf.io.parse_single_example(record, self.features_dict)
            if self._map_func is not None:
                return self._map_func(features)
            else:
                return features

        filenames = tf.data.Dataset.list_files(self._file_pattern,
                                               shuffle=shuffle)
        dataset = filenames.interleave(
            map_func=self._data_format_map_fn,
            cycle_length=40,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            deterministic=True)
        dataset = dataset.map(parse_tfexample,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE,
                              deterministic=True)
        return dataset

    def get_batch(self, batch_size, shuffle=True, repeats=-1):
        dataset = self.get_dataset(shuffle)
        dataset = dataset.repeat(repeats)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    @property
    def features_dict(self):
        return {
            'sample_name': tf.io.FixedLenFeature([1], dtype=tf.string),
            'instrument_id': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'note_number': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'velocity': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'instrument_source': tf.io.FixedLenFeature([1], dtype=tf.int64),
            'qualities': tf.io.FixedLenFeature([10], dtype=tf.int64),
            'audio': tf.io.FixedLenFeature([self._audio_length], dtype=tf.float32),
            'f0_hz': tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'f0_confidence': tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'loudness_db': tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'f0_scaled': tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'ld_scaled': tf.io.FixedLenFeature([self._feature_length], dtype=tf.float32),
            'z': tf.io.FixedLenFeature([self._feature_length * 16], dtype=tf.float32),
            'f0_estimate': tf.io.FixedLenFeature([], dtype=tf.string),
            'h_freq': tf.io.FixedLenFeature([], dtype=tf.string),
            'h_mag': tf.io.FixedLenFeature([], dtype=tf.string),
            'h_phase': tf.io.FixedLenFeature([], dtype=tf.string),
        }


def create_dataset(
        dataset_path,
        batch_size=16,
        example_secs=4,
        sample_rate=16000,
        frame_rate=250,
        map_func=None):
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

    harmonics = tf.shape(h_freq)[-1] - 2
    h_freq = h_freq[:, :, :harmonics]
    h_mag = h_mag[:, :, :harmonics]
    h_phase = h_phase[:, :, :harmonics] if h_phase is not None else h_phase

    normalized_data = conf.data_handler.normalize(
        h_freq, h_mag, h_phase, note_number)

    h_freq, h_mag, _ = conf.data_handler.denormalize(
        normalized_data, phase_mode='none')
    measures = conf.data_handler.compute_measures(h_freq, h_mag)

    for k, v in normalized_data.items():
        normalized_data[k] = tf.squeeze(v, axis=0)

    for k, v in measures.items():
        measures[k] = tf.squeeze(v, axis=0)

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

    inputs = normalized_data.copy()
    inputs.update({
        "name": name,
        "note_number": tf.squeeze(note_number),
        "velocity": tf.squeeze(velocity),
        "instrument_id": tf.squeeze(instrument_id),
        "measures": tf.stack(tf.nest.flatten(measures), axis=-1),
    })

    targets = normalized_data.copy()
    targets.update(measures)

    return inputs, targets


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

    train_dataset = create_dataset(
        train_path, map_func=map_features, batch_size=conf.batch_size)
    valid_dataset = create_dataset(
        valid_path, map_func=map_features, batch_size=conf.batch_size)
    test_dataset = create_dataset(
        test_path, map_func=map_features, batch_size=conf.batch_size)

    if conf.dataset_modifier is not None:
        train_dataset, valid_dataset, test_dataset = conf.dataset_modifier(
            train_dataset, valid_dataset, test_dataset)

    return train_dataset, valid_dataset, test_dataset
