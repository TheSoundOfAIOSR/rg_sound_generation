#!/usr/bin/env python
# coding: utf-8

"""
Run from command line:
    python causal_conv_test.py --dataset_dir ../nsynth_dataset --model_dir_name causal_conv
"""

import tensorflow as tf
import click

from tensorflow.keras.layers import Input, concatenate, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.layers import Dense, Reshape
import sys
sys.path.append('..')
from mapping_models import trainer


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
    note_number = convert_to_sequence(note_number) # 1000, 1
    velocity = convert_to_sequence(velocity)
    instrument_source = convert_to_sequence(instrument_source)
    qualities = convert_to_sequence(qualities) # 1000, 10

    f0_scaled = tf.expand_dims(f0_scaled, axis=-1)
    ld_scaled = tf.expand_dims(ld_scaled, axis=-1)
    z = tf.reshape(z, shape=(sequence_length, 16))

    inputs = {
        'pitch': note_number,
        'velocity': velocity,
        'instrument_source': instrument_source,
        'qualities': qualities,
        'latent_vector': z
    }
    
    outputs = {
        'f0_scaled': f0_scaled,
        'ld_scaled': ld_scaled
    }

    return inputs, outputs


def create_model():
    _pitch = Input(shape=(1000, 1), name='pitch')
    _velocity = Input(shape=(1000, 1), name='velocity')
    _instrument_source = Input(shape=(1000, 1), name='instrument_source')
    _qualities = Input(shape=(1000, 10), name='qualities')
    _latent_sample = Input(shape=(1000, 16), name='latent_vector')
    
    _input = concatenate([_instrument_source, _qualities, _latent_sample], axis=-1, name='concat_1')
    
    x = _input
    
    for i in range(0, 4):
        n_filters = 2**(4 + i)
        x = Conv1D(n_filters, 5, activation='relu', strides=2, padding='causal', name=f'conv_{i + 1}')(x)
        x = MaxPool1D(pool_size=2, name=f'pool_{i + 1}')(x)
    
    x = Flatten(name='flatten')(x)
    
    _pitch_x = Reshape((1000, ), name='pitch_reshaped')(_pitch)
    _pitch_x = Dense(512, activation='relu', name='pitch_dense_1')(_pitch_x)
    _f0_x = concatenate([x, _pitch_x], name='concat_f0')
    
    _velocity_x = Reshape((1000, ), name='velocity_reshaped')(_velocity)
    _velocity_x = Dense(512, activation='relu', name='velocity_dense_1')(_velocity_x)
    _ld_x = concatenate([x, _velocity_x], name='concat_ld')
    
    _f0_scaled = Dense(1000, activation='linear', name='f0_scaled')(_f0_x)
    _ld_scaled = Dense(1000, activation='linear', name='ld_scaled')(_ld_x)
    
    model = tf.keras.models.Model([_instrument_source, _qualities, _latent_sample, _velocity, _pitch],
                                  [_f0_scaled, _ld_scaled], name='cc')
    return model


@click.command()
@click.option('--dataset_dir', help='Location of root directory of the dataset')
@click.option('--model_dir_name',
              help='Name of checkpoint directory, will be created inside the main checkpoint directory')
@click.option('--epochs', default=1, help='Number of training epochs')
@click.option('--batch_size', default=64, help='Batch size')
def train(dataset_dir, model_dir_name, epochs, batch_size):
    model = create_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss='mse'
    )

    total_examples = 32690
    validation_examples = 2081
    steps = int(total_examples / batch_size)
    validation_steps = int(validation_examples / batch_size)

    trainer.train(
        model,
        dataset_dir=dataset_dir,
        model_dir=model_dir_name,
        epochs=epochs,
        features_map=features_map,
        steps_per_epoch=steps,
        validation_steps=validation_steps,
        batch_size=batch_size,
        verbose=True
    )


if __name__ == '__main__':
    train()
