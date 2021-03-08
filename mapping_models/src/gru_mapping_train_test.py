import tensorflow as tf
from data_providers import CompleteTFRecordProvider
import os


# ------------------------------------------------------------------------------

def create_dataset(dataset_dir='nsynth_guitar',
                   split='train',
                   batch_size=16,
                   example_secs=4,
                   sample_rate=16000,
                   frame_rate=250,
                   map_func=None):
    assert os.path.exists(dataset_dir)

    split_dataset_dir = os.path.join(dataset_dir, split)
    tfrecord_file = os.path.join(split_dataset_dir, 'complete.tfrecord')

    train_data_provider = CompleteTFRecordProvider(
        file_pattern=tfrecord_file + '*',
        example_secs=example_secs,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        map_func=map_func)

    dataset = train_data_provider.get_batch(
        batch_size,
        shuffle=True,
        repeats=-1)

    return dataset


# ------------------------------------------------------------------------------

def features_map(features):
    note_number = features['note_number']
    velocity = features['velocity']
    instrument_source = features['instrument_source']
    qualities = features['qualities']
    f0_scaled = features['f0_scaled']
    ld_scaled = features['ld_scaled']
    z = features['z']

    sequence_length = f0_scaled.shape[0]

    def convert_to_sequence(feature):
        channels = feature.shape[0]
        feature = tf.expand_dims(feature, axis=0)

        feature = tf.broadcast_to(feature, shape=(sequence_length, channels))
        feature = tf.cast(feature, dtype=tf.float32)

        return feature

    # Normalize data
    # 0-127
    note_number = note_number / 127
    velocity = velocity / 127

    # 0-2
    # 0	acoustic, 1	electronic, 2	synthetic
    instrument_source = instrument_source / 2

    # Prepare dataset for a sequence to sequence mapping
    note_number = convert_to_sequence(note_number)
    velocity = convert_to_sequence(velocity)
    instrument_source = convert_to_sequence(instrument_source)
    qualities = convert_to_sequence(qualities)

    f0_scaled = tf.expand_dims(f0_scaled, axis=-1)
    ld_scaled = tf.expand_dims(ld_scaled, axis=-1)
    z = tf.reshape(z, shape=(sequence_length, 16))

    inputs = tf.concat(
        [note_number, velocity, instrument_source, qualities, z],
        axis=-1)

    targets = tf.concat(
        [f0_scaled, ld_scaled],
        axis=-1)

    return inputs, targets


# ------------------------------------------------------------------------------

def train():
    # create datasets
    train_dataset = create_dataset(split='train', map_func=features_map)
    valid_dataset = create_dataset(split='valid', map_func=features_map)
    test_dataset = create_dataset(split='test', map_func=features_map)

    # create and compile mapping model
    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(32, return_sequences=True),
        tf.keras.layers.Dense(2, activation='tanh')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.losses.MeanSquaredError()])

    # build model
    x_train, y_train = next(iter(train_dataset))
    _ = model(x_train)

    print(model.summary())

    # load model checkpoint
    checkpoint_dir = 'checkpoints'
    checkpoint_file = os.path.join(checkpoint_dir, 'cp.ckpt')

    if os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir):
        model.load_weights(checkpoint_file)

    # create training callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_weights_only=True,
        verbose=0,
        save_freq='epoch')

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=5)

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.9

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # train model
    epochs = 20
    steps_per_epoch = 100
    validation_steps = 10

    model.fit(train_dataset,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_dataset,
              validation_steps=validation_steps,
              callbacks=[checkpoint, early_stop, lr_scheduler])

    # evaluate model
    model.evaluate(test_dataset, steps=500)


# ------------------------------------------------------------------------------

if __name__ == '__main__':
    train()
