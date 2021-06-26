import tensorflow as tf
import ddsp.training.data as data


class CompleteTFRecordProvider(data.RecordProvider):
    def __init__(self,
                 file_pattern=None,
                 example_secs=4,
                 sample_rate=16000,
                 frame_rate=250,
                 map_func=None):
        super().__init__(file_pattern, example_secs, sample_rate,
                         frame_rate, tf.data.TFRecordDataset)
        self._map_func = map_func

    def get_dataset(self, shuffle=True):
      def parse_tfexample(record):
        features = tf.io.parse_single_example(record, self.features_dict)
        if self._map_func is not None:
          return self._map_func(features)
        else:
          return features

      filenames = tf.data.Dataset.list_files(self._file_pattern, shuffle=shuffle)
      dataset = filenames.interleave(
          map_func=self._data_format_map_fn,
          cycle_length=40,
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=True)
      dataset = dataset.map(parse_tfexample,
                            num_parallel_calls=tf.data.experimental.AUTOTUNE,
                            deterministic=True)
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
