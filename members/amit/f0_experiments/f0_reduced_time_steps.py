import tensorflow as tf
import os
import click

from tensorflow.keras.layers import Input, concatenate, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.layers import Dense, Dropout, RepeatVector
from tfrecord_provider import CompleteTFRecordProvider
from ddsp.core import midi_to_unit


def create_model():
    _pitch = Input(shape=(1, ), name='pitch')
    _velocity = Input(shape=(5, ), name='velocity')
    _instrument_source = Input(shape=(3, ), name='instrument_source')
    _qualities = Input(shape=(10, ), name='qualities')
    _latent_sample = Input(shape=(1000, 16), name='latent_vector')

    r_pitch = RepeatVector(1000, name='repeat_pitch')(_pitch)
    r_velocity = RepeatVector(1000, name='repeat_velocity')(_velocity)
    r_instrument_source = RepeatVector(1000, name='repeat_instrument_source')(_instrument_source)
    r_qualities = RepeatVector(1000, name='repeat_qualities')(_qualities)

    _input = concatenate([r_instrument_source, r_qualities, _latent_sample, r_velocity, r_pitch], axis=-1, name='concat_1')

    x = _input

    for i in range(0, 4):
        n_filters = 2 ** (5 + i)
        x = Conv1D(n_filters, 5, activation='relu', strides=2, padding='causal', name=f'conv_{i + 1}')(x)
        x = MaxPool1D(pool_size=2, name=f'pool_{i + 1}')(x)

    x = Flatten(name='flatten')(x)

    for i in range(0, 2):
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.25)(x)

    _f0_scaled = Dense(500, activation='linear', name='f0_scaled')(x)
    _ld_scaled = Dense(1000, activation='linear', name='ld_scaled')(x)

    model = tf.keras.models.Model([_instrument_source, _qualities, _latent_sample, _velocity, _pitch],
                                  [_f0_scaled, _ld_scaled], name='f0_reduced_model')
    model.compile(
        loss='mse',
        optimizer='adam'
    )
    return model


def data_transformation(features):
    """
    Iteration 2:
        Reduced time steps to exclude first 8 ms and last 2 seconds
        Pitch as categorical input
        Clip f0_scaled to +/- 1 semitone
            This will require finding the +1 and -1 pitch range
            for each MIDI pitch in f0_scaled scale
    """
    note_number = tf.cast(features['note_number'], tf.float32)
    pitch = midi_to_unit(note_number, clip=True, midi_min=0, midi_max=127)
    velocity = (features['velocity'] - 25) / 25
    velocity_categorical = tf.keras.utils.to_categorical(velocity, num_classes=5)
    instrument_source = tf.keras.utils.to_categorical(features['instrument_source'], num_classes=3)
    qualities = features['qualities']
    ld_scaled = features['ld_scaled']
    # ignoring the last 2 seconds
    f0_scaled = features['f0_scaled'][:, :500]
    # clip f0_scaled between two octaves
    f0_scaled_max = midi_to_unit(note_number + 2., clip=True, midi_min=0, midi_max=127)
    f0_scaled_min = midi_to_unit(note_number - 2., clip=True, midi_min=0, midi_max=127)
    f0_scaled = tf.clip_by_value(f0_scaled, f0_scaled_min, f0_scaled_max)
    batch_size = features['z'].shape[0]

    inputs = {
        'pitch': pitch,
        'velocity': velocity_categorical,
        'instrument_source': instrument_source,
        'qualities': qualities,
        'latent_vector': tf.reshape(features['z'], (batch_size, 1000, 16))
    }

    outputs = {
        'f0_scaled': f0_scaled,
        'ld_scaled': ld_scaled
    }
    return inputs, outputs


def data_generator_io(data_iterable):
    while True:
        yield data_transformation(next(data_iterable))


def data_generator_iof(data_iterable):
    while True:
        features = next(data_iterable)
        yield data_transformation(features), features


@click.command()
@click.option('--dataset_dir')
@click.option('--resume_from')
@click.option('--batch_size', default=32)
@click.option('--epochs', default=100)
def train(dataset_dir, resume_from, batch_size, epochs):
    data_iterables = {}

    click.echo('creating data iterables..')

    for set_name in ['train', 'valid']:
        tfrecord_path = os.path.join(dataset_dir, set_name, 'complete.tfrecord')
        data_iterables[set_name] = iter(CompleteTFRecordProvider(tfrecord_path).get_batch(batch_size=batch_size))
        assert os.path.isfile(tfrecord_path), f'No tfrecord found at {tfrecord_path}'

    click.echo('creating model..')

    model = create_model()
    print(model.summary())

    if resume_from is not None:
        click.echo(f'Resuming training from checkpoint {resume_from}')
        model.load_weights(resume_from)

    steps = int(32690 / batch_size)
    validation_steps = int(2090 / batch_size)

    click.echo('starting training..')

    _ = model.fit(
        data_generator_io(data_iterables['train']),
        validation_data=data_generator_io(data_iterables['valid']),
        steps_per_epoch=steps,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8),
            tf.keras.callbacks.ModelCheckpoint(
                'checkpoints/f0_clipped_val_loss_{val_loss:.4f}.h5',
                save_best_only=True, monitor='val_loss'
            ),
            tf.keras.callbacks.CSVLogger(
                'logs/f0_clipped.csv', separator=",",
                append=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4)
        ]
    )

    click.echo('training finished..')


if __name__ == '__main__':
    train()
