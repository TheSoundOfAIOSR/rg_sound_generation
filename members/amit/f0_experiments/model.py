import tensorflow as tf

from tensorflow.keras.layers import Input, concatenate, RepeatVector
from tensorflow.keras.layers import Dense, GRU, Dropout


def create_model():
    _velocity = Input(shape=(5,), name='velocity')
    _instrument_source = Input(shape=(3,), name='instrument_source')
    _qualities = Input(shape=(10,), name='qualities')
    _z = Input(shape=(1000, 16), name='z')

    categorical_inputs = concatenate(
        [_velocity, _instrument_source, _qualities],
        name='categorical_inputs'
    )
    _input = concatenate(
        [_z, RepeatVector(1000, name='repeat')(categorical_inputs)],
        name='total_inputs'
    )

    x = GRU(256, return_sequences=True, name='gru_1')(_input)
    x = Dropout(0.5, name='dropout_1')(x)
    x = GRU(256, return_sequences=True, name='gru_2')(x)
    x = Dropout(0.5, name='dropout_2')(x)

    _f0_categorical = Dense(49, activation='softmax', name='f0_categorical')(x)

    model = tf.keras.models.Model(
        [_velocity, _instrument_source, _qualities, _z],
        _f0_categorical
    )

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
