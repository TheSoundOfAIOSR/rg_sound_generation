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
    x = Conv1D(num_features, 64, padding='same', activation='relu')(x)
    x = Dense(num_features, activation='sigmoid', name='decoder_output')(x)

    return tf.keras.models.Model(
        _input, x,
        name='decoder'
    )


class VAE(tf.keras.models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z, z_mean, z_log_var = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, reconstruction), axis=1)
        )
        kl_loss = tf.reduce_mean(
            tf.reduce_sum(-0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)), axis=1)
        )
        total_loss = reconstruction_loss + kl_loss
        self.add_loss(total_loss)
        return reconstruction
