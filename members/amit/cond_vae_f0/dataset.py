# from f0_scaled to f0_scaled conditioned on instrument_id

import os
import tensorflow as tf
from tfrecord_provider import CompleteTFRecordProvider
from config import *


with open("instrument_id_map.csv", "r") as f:
    rows = f.read().splitlines()
    names = rows[0].split(",")
    ids = [int(x) for x in rows[1].split(",")]
    keys_tensor = tf.constant(names)
    values_tensor = tf.constant(ids)
    init = tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor)

num_instruments = len(ids)
instrument_to_ids = tf.lookup.StaticHashTable(init, default_value=-1)


def create_dataset(
        dataset_dir='nsynth_guitar',
        split='train',
        batch_size=16,
        example_secs=4,
        sample_rate=16000,
        frame_rate=250,
        map_func=None
):
    assert os.path.exists(dataset_dir)

    split_dataset_dir = os.path.join(dataset_dir, split)
    tfrecord_file = os.path.join(split_dataset_dir, 'complete.tfrecord')

    train_data_provider = CompleteTFRecordProvider(
        file_pattern=tfrecord_file + '*',
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        map_func=map_func
    )

    dataset = train_data_provider.get_batch(
        batch_size,
        shuffle=True,
        repeats=-1
    )

    return dataset


def map_features(features):
    f0_scaled = features['f0_scaled']
    sample_name = tf.squeeze(features['sample_name'])
    note_number = features['note_number']
    note_number = tf.squeeze(tf.one_hot(note_number, depth=num_pitches))
    instrument = tf.strings.substr(sample_name, pos=7, len=tf.strings.length(sample_name) - 15)
    index = instrument_to_ids.lookup(instrument)
    instrument_id = tf.squeeze(tf.one_hot(index, depth=num_instruments))
    return {
        'f0_scaled': f0_scaled,
        'note_number': note_number,
        'instrument_id': instrument_id
    }


dataset_dir = "D:\soundofai\\complete_data"

train_dataset = create_dataset(dataset_dir, split="train",
                               map_func=map_features, batch_size=batch_size)
valid_dataset = create_dataset(dataset_dir, split="valid",
                               map_func=map_features, batch_size=batch_size)
