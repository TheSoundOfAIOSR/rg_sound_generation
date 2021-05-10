from typing import Dict
import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate, Conv1D, MaxPool1D, Flatten, Dense, Dropout
from loguru import logger


class F0LoudnessGenerator:
    def __init__(self, checkpoint_path):
        self.model = None
        self.checkpoint_path = checkpoint_path
        self._load_model()

    def _load_model(self):
        _pitch = Input(shape=(1000, 1), name='pitch')
        _velocity = Input(shape=(1000, 1), name='velocity')
        _instrument_source = Input(shape=(1000, 1), name='instrument_source')
        _qualities = Input(shape=(1000, 10), name='qualities')
        _latent_sample = Input(shape=(1000, 16), name='latent_vector')

        _input = concatenate(
            [_instrument_source, _qualities, _latent_sample, _velocity, _pitch],
            axis=-1, name='concat_1'
        )

        x = _input

        for i in range(0, 4):
            n_filters = 2 ** (5 + i)
            x = Conv1D(
                n_filters, 5, activation='relu',
                strides=2, padding='causal',
                name=f'conv_{i + 1}'
            )(x)
            x = MaxPool1D(pool_size=2, name=f'pool_{i + 1}')(x)

        x = Flatten(name='flatten')(x)

        for i in range(0, 2):
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.25)(x)

        _f0_scaled = Dense(1000, activation='linear', name='f0_scaled')(x)
        _ld_scaled = Dense(1000, activation='linear', name='ld_scaled')(x)

        self.model = tf.keras.models.Model(
            [_instrument_source, _qualities, _latent_sample, _velocity, _pitch],
            [_f0_scaled, _ld_scaled], name='cc'
        )
        logger.info("Loading model checkpoint")
        self.model.load_weights(self.checkpoint_path)
        self.model.trainable = False

    def predict(self, inputs: Dict) -> tf.Tensor:
        logger.info("Fetching prediction")
        return self.model.predict(inputs)
