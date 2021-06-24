import os
from tfrecord_provider import CompleteTFRecordProvider
from features import features_map


def get_data(dataset_dir: str):
    train_tfrecord_file = os.path.join(dataset_dir, 'train.tfrecord')
    valid_tfrecord_file = os.path.join(dataset_dir, 'valid.tfrecord')
    test_tfrecord_file = os.path.join(dataset_dir, 'test.tfrecord')

    example_secs = 4
    sample_rate = 16000
    frame_rate = 250

    # Create train dataset
    train_data_provider = CompleteTFRecordProvider(
        file_pattern=train_tfrecord_file,
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        map_func=features_map)

    train_dataset = train_data_provider.get_dataset(shuffle=False)

    # Create valid dataset
    valid_data_provider = CompleteTFRecordProvider(
        file_pattern=valid_tfrecord_file,
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        map_func=features_map)

    valid_dataset = valid_data_provider.get_dataset(shuffle=False)

    # Create test dataset
    test_data_provider = CompleteTFRecordProvider(
        file_pattern=test_tfrecord_file,
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        map_func=features_map)

    test_dataset = test_data_provider.get_dataset(shuffle=False)

    return train_dataset, valid_dataset, test_dataset
