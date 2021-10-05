import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import random
from create_dataset import prepare_example
from read_dataset import TFRecordProvider


def split_dataset(source_dir, target_dir):
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    train_pattern = os.path.join(source_dir, "train.tfrecord")
    valid_pattern = os.path.join(source_dir, "valid.tfrecord")
    test_pattern = os.path.join(source_dir, "test.tfrecord")

    train_target_path = os.path.join(target_dir, "train.tfrecord")
    valid_target_path = os.path.join(target_dir, "valid.tfrecord")

    train_dp = TFRecordProvider(train_pattern)
    valid_dp = TFRecordProvider(valid_pattern)
    test_dp = TFRecordProvider(test_pattern)

    dataset = train_dp.get_batch(batch_size=1, shuffle=False, repeats=1)
    valid_dataset = valid_dp.get_batch(1, shuffle=False, repeats=1)
    test_dataset = test_dp.get_batch(1, shuffle=False, repeats=1)

    dataset.concatenate(valid_dataset)
    dataset.concatenate(test_dataset)

    train_writer = tf.io.TFRecordWriter(train_target_path)
    valid_writer = tf.io.TFRecordWriter(valid_target_path)

    train_count, valid_count = 0, 0

    for e in tqdm(dataset):
        sample_name = e["sample_name"][0].numpy()
        instrument_id = e["instrument_id"][0].numpy()
        note_number = e['note_number'][0].numpy()
        velocity = e['velocity'][0].numpy()
        audio = e['audio'][0].numpy()
        h_freq = e["h_freq"][0].numpy()
        h_mag = e["h_mag"][0].numpy()
        h_phase = e["h_phase"][0].numpy()

        # we can ignore velocity 25 for id 49
        if np.squeeze(instrument_id) == 49 and np.squeeze(velocity) == 25:
            continue

        target_set = random.randint(0, 99)

        tf_example = prepare_example(sample_name, note_number, velocity,
                                     instrument_id, audio, h_freq, h_mag, h_phase)

        if target_set < 10:
            valid_count += 1
            valid_writer.write(tf_example.SerializeToString())
        else:
            train_count += 1
            train_writer.write(tf_example.SerializeToString())

    train_writer.close()
    valid_writer.close()

    print(f"Finished with num training examples: {train_count}, num validation examples: {valid_count}")
    print(f"The {source_dir} can now be deleted to save space, if required")
