import tensorflow as tf

from tensorflow.keras.layers import Conv1D, MaxPool1D, Flatten, Dense, Input
from tensorflow.keras.layers import UpSampling1D, Reshape
from layers import SamplingLayer


def create_encoder(latent_dim, num_features):
    def conv_block(inputs, filters, kernel_size, strides):
        x = Conv1D(filters, kernel_size, strides=strides, padding='same', activation='relu')(inputs)
        return MaxPool1D(2)(x)

    _input = Input(shape=(1024, num_features), name='encoder_input')

    x = conv_block(_input, 1024, 512, 4)

    for filters in [128] * 3 + [256, 512]:
        x = conv_block(x, filters, 64, 1)

    x = Flatten()(x)

    z_mean = Dense(latent_dim, activation='relu', name='z_mean')(x)
    z_log_variance = Dense(latent_dim, activation='relu', name='z_log_variance')(x)
    z = SamplingLayer()([z_mean, z_log_variance])

    return tf.keras.models.Model(
        _input, [z, z_mean, z_log_variance],
        name='encoder'
    )


def create_decoder(latent_dim, num_features):
    def up_sample_block(inputs, filters, kernel_size, strides):
        x = UpSampling1D(2)(inputs)
        return Conv1D(filters, kernel_size, strides=strides, padding='same', activation='relu')(x)

    _input = Input(shape=(latent_dim, ), name='decoder_input')

    x = Dense(3 * 512, activation='relu')(_input)
    x = Reshape((3, 512))(x)

    for filters in [256, 512] + [128] * 3:
        x = up_sample_block(x, filters, 64, 1)

    x = UpSampling1D(2)(x)
    x = Reshape((1024, 24))(x)
    x = Conv1D(num_features, 64, padding='same', activation='sigmoid', name='decoder_output')(x)

    return tf.keras.models.Model(
        _input, x,
        name='decoder'
    )


def create_vae(latent_dim, num_features):
    encoder = create_encoder(latent_dim, num_features)
    decoder = create_decoder(latent_dim, num_features)

    _input = Input(shape=(1024, 16), name='vae_input')

    z, _, _ = encoder(_input)
    _output = decoder(z)

    return tf.keras.models.Model(_input, _output, name='vae')
