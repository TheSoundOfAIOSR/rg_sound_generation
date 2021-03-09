"""
Run from command line:
    python gru_test_model.py --dataset_dir ../nsynth_dataset --model_dir_name gru
"""

import tensorflow as tf
import click
import sys


def features_map(features):
    def convert_to_sequence(feature):
        channels = feature.shape[0]
        feature = tf.expand_dims(feature, axis=0)

        feature = tf.broadcast_to(feature, shape=(sequence_length, channels))
        feature = tf.cast(feature, dtype=tf.float32)

        return feature

    note_number = features['note_number']
    velocity = features['velocity']
    instrument_source = features['instrument_source']
    qualities = features['qualities']
    f0_scaled = features['f0_scaled']
    ld_scaled = features['ld_scaled']
    z = features['z']

    sequence_length = f0_scaled.shape[0]

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

    inputs = tf.concat([note_number, velocity, instrument_source, qualities, z], axis=-1)
    targets = tf.concat([f0_scaled, ld_scaled], axis=-1)

    return inputs, targets


@click.command()
@click.option('--dataset_dir', help='Location of root directory of the dataset')
@click.option('--model_dir_name',
              help='Name of checkpoint directory, will be created inside the main checkpoint directory')
@click.option('--epochs', default=1, help='Number of training epochs')
def train(dataset_dir, model_dir_name, epochs):
    sys.path.append('..')
    import trainer

    model = tf.keras.models.Sequential([
        tf.keras.layers.GRU(32, return_sequences=True),
        tf.keras.layers.Dense(2, activation='tanh')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=[tf.keras.losses.MeanSquaredError()]
    )

    trainer.train(
        model,
        dataset_dir=dataset_dir,
        model_dir=model_dir_name,
        epochs=epochs,
        features_map=features_map
    )


if __name__ == '__main__':
    train()
