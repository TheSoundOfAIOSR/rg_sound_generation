import tensorflow as tf

from tensorflow.keras.layers import Dense, Reshape, Input, Conv1D, concatenate
from tensorflow.keras.layers import BatchNormalization, UpSampling1D, Activation


class ZGenerator:
    def __init__(self, checkpoint_path):
        self.model = None
        self.checkpoint_path = checkpoint_path
        self._load_model()

    def _load_model(self):
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

        self.model = tf.keras.models.Model([_instrument_source, _qualities, _z_input, _velocity, _pitch], _output,
                                      name='z_generator')
        self.model.load_weights(self.checkpoint_path)
        self.model.trainable = False

    def predict(self, inputs):
        return self.model.predict(inputs)
