import tensorflow as tf
from config import Config


def conditioning_function(sample):
    return sample['note_number'], sample['instrument_id'], sample['velocity']


def return_conditioning(dataset):
    return dataset.map(lambda x: conditioning_function(x),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                       deterministic=True)


def pad_function(sample):
    return tf.pad(sample,
                  tf.convert_to_tensor([[0, 0], [0, Config.max_num_harmonics - tf.shape(sample)[1]]]))


def pad_dataset(dataset):
    return dataset.map(lambda x: pad_function(x),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE,
                       deterministic=True)


def filter_instruments(sample):
    banned_ids = tf.constant([6, 11, 13, 19, 23, 25, 30, 48, 49, 51, 64, 71, 79, 80, 82, 90, 92])
    instrument_id = sample['instrument_id']
    isbanned = tf.equal(banned_ids, tf.cast(instrument_id, banned_ids.dtype))
    reduced = tf.reduce_sum(tf.cast(isbanned, tf.float32))
    return tf.equal(reduced, tf.constant(0.))


def filter_pitches(sample):
    pitches = tf.constant([x for x in range(40, 89)])
    note_number = sample['note_number']
    isvalid = tf.equal(pitches, tf.cast(note_number, pitches.dtype))
    reduced = tf.reduce_sum(tf.cast(isvalid, tf.float32))
    return tf.greater(reduced, tf.constant(0.))


def input_preprocessing(dataset):
    # source dataset is now already filtered with right pitches and instrument IDs
    # dataset_filter = dataset.filter(filter_instruments).filter(filter_pitches)

    h_freq_centered = dataset.map(lambda x: x['h_freq_centered'])
    h_mag = dataset.map(lambda x: x['h_mag'])
    d_phase = dataset.map(lambda x: x['d_phase'])
    h_freq_norm = dataset.map(lambda x: x['h_freq_norm'])
    h_mag_norm = dataset.map(lambda x: x['h_mag_norm'])
    d_phase_norm = dataset.map(lambda x: x['d_phase_norm'])

    h_freq_pad = pad_dataset(h_freq_centered)
    h_mag_pad = pad_dataset(h_mag)
    d_phase_pad = pad_dataset(d_phase)
    h_freq_norm_pad = pad_dataset(h_freq_norm)
    h_mag_norm_pad = pad_dataset(h_mag_norm)
    d_phase_norm_pad = pad_dataset(d_phase_norm)

    conditioning_dataset = return_conditioning(dataset)

    return tf.data.Dataset.zip((h_freq_norm_pad, h_mag_norm_pad, d_phase_norm_pad,
                                h_freq_pad, h_mag_pad, d_phase_pad,
                                conditioning_dataset))
