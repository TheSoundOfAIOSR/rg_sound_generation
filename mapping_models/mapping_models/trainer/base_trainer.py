import tensorflow as tf
import os

from mapping_models.data_providers import CompleteTFRecordProvider


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


def get_callbacks(checkpoint_file, min_lr=1e-7):
    def scheduler(epoch, lr):
        if (epoch + 1) % 5 == 0:
            return max(lr * 0.9, min_lr)
        return lr

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_weights_only=True,
        save_best_only=True,
        verbose=0,
        save_freq='epoch'
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    return [checkpoint, early_stop, lr_scheduler]


def train(
        model,
        dataset_dir,
        model_dir,
        features_map=None,
        epochs=20,
        steps_per_epoch=100,
        validation_steps=10,
        batch_size=16,
        verbose=2,
        load_checkpoint=False
):
    # create datasets
    train_dataset = create_dataset(dataset_dir=dataset_dir, split='train',
                                   map_func=features_map, batch_size=batch_size)
    valid_dataset = create_dataset(dataset_dir=dataset_dir, split='valid',
                                   map_func=features_map, batch_size=batch_size)
    test_dataset = create_dataset(dataset_dir=dataset_dir, split='test',
                                  map_func=features_map, batch_size=batch_size)

    # build model
    x_train, y_train = next(iter(train_dataset))
    _ = model(x_train)

    print(model.summary())

    checkpoint_dir = os.path.join(os.getcwd(), 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, model_dir, 'cp.ckpt')

    if os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir) and load_checkpoint:
        try:
            model.load_weights(checkpoint_file)
        except Exception as e:
            print(e)

    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_dataset,
        validation_steps=validation_steps,
        callbacks=get_callbacks(checkpoint_file),
        verbose=verbose
    )

    # evaluate model
    model.evaluate(test_dataset, steps=10)
