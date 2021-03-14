#!/usr/bin/env python
# coding: utf-8

"""
Run from command line:
    python z_generator.py --dataset_dir ../nsynth_dataset --model_dir_name z_gen
"""

import tensorflow as tf
import click

from tensorflow.keras.layers import Input, concatenate, Conv1D, MaxPool1D, UpSampling1D
from tensorflow.keras.layers import BatchNormalization, Reshape, Activation, Dense
from mapping_models import trainer


def features_map_decoder(features):
    note_number = features['note_number']
    velocity = features['velocity']
    instrument_source = features['instrument_source']
    qualities = features['qualities']
    z = features['z']

    z_output = tf.reshape(z, shape=(1000, 16))
    z_input = z_output[0, :]

    # Normalize data
    # 0-127
    note_number = note_number / 127
    velocity = velocity / 127

    # 0-2
    # 0	acoustic, 1	electronic, 2	synthetic
    instrument_source = instrument_source / 2

    # latent_vector is not used in the training, just for inference of f0, ld
    inputs = {
        'pitch': note_number,
        'velocity': velocity,
        'instrument_source': instrument_source,
        'qualities': qualities,
        'z_input': z_input,
        'latent_vector': z_output
    }

    outputs = {
        'z_output': z_output
    }

    return inputs, outputs


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
    z_output = tf.reshape(z, shape=(sequence_length, 16))
    z_input = z_output[0, :]

    # Normalize data
    # 0-127
    note_number = note_number / 127
    velocity = velocity / 127

    # 0-2
    # 0	acoustic, 1	electronic, 2	synthetic
    instrument_source = instrument_source / 2

    # Prepare dataset for a sequence to sequence mapping
    note_number = convert_to_sequence(note_number)  # 1000, 1
    velocity = convert_to_sequence(velocity)
    instrument_source = convert_to_sequence(instrument_source)
    qualities = convert_to_sequence(qualities)  # 1000, 10
    z_input = convert_to_sequence(z_input)

    # f0_scaled = tf.expand_dims(f0_scaled, axis=-1)
    # ld_scaled = tf.expand_dims(ld_scaled, axis=-1)

    # latent_vector is not used in the training, just for inference of f0, ld
    inputs = {
        'pitch': note_number,
        'velocity': velocity,
        'instrument_source': instrument_source,
        'qualities': qualities,
        'z_input': z_input,
        'latent_vector': z_output
    }

    outputs = {
        'z_output': z_output
    }

    return inputs, outputs


def create_model_decoder():
    _pitch = Input(shape=(1, ), name='pitch')
    _velocity = Input(shape=(1, ), name='velocity')
    _instrument_source = Input(shape=(1, ), name='instrument_source')
    _qualities = Input(shape=(10, ), name='qualities')
    _z_input = Input(shape=(16, ), name='z_input')

    # there is no causality in this input, the temporal dimension is just for convenience
    _input = concatenate([_instrument_source, _qualities, _z_input, _velocity, _pitch], axis=-1, name='concat_1')

    x = Dense(256, activation='relu', name='dense_1')(_input)
    x = Reshape((1, 256), name='reshape_1')(x)

    for i in range(0, 5):
        n_filters = 2 ** (8 - i)
        x = UpSampling1D(2, name=f'up_{i}')(x)
        x = Conv1D(n_filters, 7, padding='causal', name=f'up_conv_{i}')(x)
        x = BatchNormalization(name=f'up_bn_{i}')(x)
        x = Activation('relu', name=f'up_act_{i}')(x)

    x = Reshape((16, 32), name='reshape_2')(x)
    x = Dense(1000, activation='linear', name='dense_2')(x)
    _output = Reshape((1000, 16), name='z_output')(x)

    model = tf.keras.models.Model([_instrument_source, _qualities, _z_input, _velocity, _pitch], _output,
                                  name='z_generator')
    return model


def create_model_full_conv():
    _pitch = Input(shape=(1000, 1), name='pitch')
    _velocity = Input(shape=(1000, 1), name='velocity')
    _instrument_source = Input(shape=(1000, 1), name='instrument_source')
    _qualities = Input(shape=(1000, 10), name='qualities')
    _z_input = Input(shape=(1000, 16), name='z_input')

    # there is no causality in this input, the temporal dimension is just for convenience
    _input = concatenate([_instrument_source, _qualities, _z_input, _velocity, _pitch], axis=-1, name='concat_1')

    x = _input

    for i in range(0, 5):
        n_filters = 2 ** (5 + i)
        x = Conv1D(n_filters, 7, strides=2, name=f'conv_{i + 1}')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if (i + 1) % 2 == 0:
            x = MaxPool1D(pool_size=2, name=f'pool_{i + 1}')(x)

    x = Reshape((2048, 1))(x)
    x = Conv1D(16, 7, activation='relu')(x)
    x = Reshape((16, 2042))(x)
    x = Dense(1000)(x)
    _z_output = Reshape((1000, 16), name='z_output')(x)

    model = tf.keras.models.Model([_instrument_source, _qualities, _z_input, _velocity, _pitch],
                                  _z_output, name='z_generator')
    return model


def create_model(model_type):
    assert model_type in ['decoder', 'full_conv']
    if model_type == 'decoder':
        return create_model_decoder()
    return create_model_full_conv()


@click.command()
@click.option('--dataset_dir', help='Location of root directory of the dataset')
@click.option('--model_dir_name',
              help='Name of checkpoint directory, will be created inside the main checkpoint directory')
@click.option('--epochs', default=1, help='Number of training epochs')
@click.option('--batch_size', default=64, help='Batch size')
@click.option('--model_type', default='decoder', help='Type of model')
def train(dataset_dir, model_dir_name, epochs, batch_size, model_type):
    model = create_model(model_type)
    tf.keras.utils.plot_model(model, show_shapes=True, to_file='images/z_generator.png')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='mse'
    )

    total_examples = 32690
    validation_examples = 2081
    steps = int(total_examples / batch_size)
    validation_steps = int(validation_examples / batch_size)

    map_func = features_map_decoder if model_type == 'decoder' else features_map

    trainer.train(
        model,
        dataset_dir=dataset_dir,
        model_dir=model_dir_name,
        epochs=epochs,
        features_map=map_func,
        steps_per_epoch=steps,
        validation_steps=validation_steps,
        batch_size=batch_size,
        verbose=True
    )


if __name__ == '__main__':
    train()
