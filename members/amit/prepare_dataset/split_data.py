import os
import numpy as np
import tensorflow as tf
import random
from prepare import CompleteTFRecordProvider
from prepare import _byte_feature, _int64_feature, _float_feature, _tensor_feature
from tqdm import tqdm


base_dir = "D:\soundofai\\nsynth-guitar-subset"

target_train_path = "D:\soundofai\\cleaned_nsynth\\train.tfrecord"
target_valid_path = "D:\soundofai\\cleaned_nsynth\\valid.tfrecord"
target_test_path = "D:\soundofai\\cleaned_nsynth\\test.tfrecord"

train_pattern = os.path.join(base_dir, "train", "complete.tfrecord")
valid_pattern = os.path.join(base_dir, "valid", "complete.tfrecord")
test_pattern = os.path.join(base_dir, "test", "complete.tfrecord")

train_dp = CompleteTFRecordProvider(train_pattern)
valid_dp = CompleteTFRecordProvider(valid_pattern)
test_dp = CompleteTFRecordProvider(test_pattern)

dataset = train_dp.get_batch(1, shuffle=False, repeats=1)
valid_dataset = valid_dp.get_batch(1, shuffle=False, repeats=1)
test_dataset = test_dp.get_batch(1, shuffle=False, repeats=1)

dataset.concatenate(valid_dataset)
dataset.concatenate(test_dataset)

train_writer = tf.io.TFRecordWriter(target_train_path)
valid_writer = tf.io.TFRecordWriter(target_valid_path)
test_writer = tf.io.TFRecordWriter(target_test_path)

train_count, valid_count, test_count = 0, 0, 0


for e in tqdm(dataset):
    sample_name = e["sample_name"][0].numpy()
    instrument_id = e["instrument_id"][0].numpy()
    note_number = e['note_number'][0].numpy()
    velocity = e['velocity'][0].numpy()
    instrument_source = e['instrument_source'][0].numpy()
    qualities = e['qualities'][0].numpy()
    audio = e['audio'][0].numpy()
    f0_hz = e['f0_hz'][0].numpy()
    f0_confidence = e['f0_confidence'][0].numpy()
    loudness_db = e['loudness_db'][0].numpy()
    f0_scaled = e["f0_scaled"][0].numpy()
    ld_scaled = e["ld_scaled"][0].numpy()
    z = e["z"][0].numpy()
    f0_estimate = e["f0_estimate"][0].numpy()
    h_freq = e["h_freq"][0].numpy()
    h_mag = e["h_mag"][0].numpy()
    h_phase = e["h_phase"][0].numpy()

    # we can ignore velocity 25 for id 49
    if np.squeeze(instrument_id) == 49 and np.squeeze(velocity) == 25:
        continue

    complete_dataset_dict = {
        'sample_name': _byte_feature(sample_name),
        'instrument_id': _int64_feature(instrument_id),
        'note_number': _int64_feature(note_number),
        'velocity': _int64_feature(velocity),
        'instrument_source': _int64_feature(instrument_source),
        'qualities': _int64_feature(qualities),
        'audio': _float_feature(audio),
        'f0_hz': _float_feature(f0_hz),
        'f0_confidence': _float_feature(f0_confidence),
        'loudness_db': _float_feature(loudness_db),
        'f0_scaled': _float_feature(f0_scaled),
        'ld_scaled': _float_feature(ld_scaled),
        'z': _float_feature(z),
        'f0_estimate': _tensor_feature(f0_estimate),
        'h_freq': _tensor_feature(h_freq),
        'h_mag': _tensor_feature(h_mag),
        'h_phase': _tensor_feature(h_phase)
    }

    tf_example = tf.train.Example(features=tf.train.Features(feature=complete_dataset_dict))

    target_set = random.randint(0, 99)

    if target_set < 70:
        train_writer.write(tf_example.SerializeToString())
        train_count += 1
    elif target_set < 90:
        valid_writer.write(tf_example.SerializeToString())
        valid_count += 1
    else:
        test_writer.write(tf_example.SerializeToString())
        test_count += 1


train_writer.close()
valid_writer.close()
test_writer.close()


print("Finished..")
print(f"Train count; {train_count}, Valid count: {valid_count}, test count: {test_count}")
