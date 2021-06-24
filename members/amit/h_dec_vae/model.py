import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import math
from config import Config


class Sampling(layers.Layer):
  """Uses (z_mean, z_log_var) to sample z, the vector encoding the input."""

  def call(self, inputs):
      z_mean, z_log_var = inputs
      batch = tf.shape(z_mean)[0]
      dim = tf.shape(z_mean)[1]
      epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
      return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        x = tf.reshape(tf.transpose(tf.stack(data[:3], axis=0), perm=[1, 2, 0]), [-1,1001,Config.max_num_harmonics,3])
        y = tf.reshape(tf.transpose(tf.stack(data[3:6], axis=0), perm=[1, 2, 0]), [-1,1001,Config.max_num_harmonics,3])
        conditioning = data[6]
        pitch = conditioning[0]
        instrument = conditioning[1]
        velocity = conditioning[2]
        mag = data[4]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder([z, pitch, instrument, velocity]) #qualities
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.math.multiply(
                        keras.losses.mean_squared_error(y, reconstruction),
                        mag),
                    axis=(1,2))
                )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def build_encoder(latent_dim, lstm_dim, units=(32, 32, 64, 64), kernel_sizes=(3, 3, 3, 3), strides=(2, 2, 2, 2)):
    encoder_inputs = keras.Input(shape=(1001, Config.max_num_harmonics, 3))
    for i, (unit, kernel_size, stride) in enumerate(zip(units, kernel_sizes, strides)):
        if i == 0:
            x = layers.Conv2D(unit, (kernel_size), activation="relu", strides=(stride), padding="same")(encoder_inputs)
        else:
            x = layers.Conv2D(unit, (kernel_size), activation="relu", strides=(stride), padding="same")(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    # x = layers.TimeDistributed(layers.Dense(lstm_dim, activation="relu"))(x)
    # x = layers.Bidirectional(layers.LSTM(lstm_dim, activation="tanh", return_sequences=True, dropout=0.1))(x)
    x = layers.LSTM(lstm_dim, activation="relu", return_sequences=False, dropout=0.1)(x)
    z_mean = layers.Dense(latent_dim, activation="relu", name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, activation="relu", name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    return encoder


def _conv_shape(strides, dim_size=(1001, Config.max_num_harmonics, 3)):
    for i in strides:
        dim_size = [math.ceil(x / i) for x in dim_size]
    return dim_size


def build_decoder(latent_dim, lstm_dim, units=(32, 32, 64, 64), kernel_sizes=(3, 3, 3, 3), strides=(2, 2, 2, 2)):
    conv_shape = _conv_shape(strides)
    units.reverse()
    kernel_sizes.reverse()
    strides.reverse()

    latent_inputs = keras.Input(shape=(latent_dim,))
    pitch_inputs = keras.Input(shape=(1,))
    instrument_inputs = keras.Input(shape=(1,))
    velocity_inputs = keras.Input(shape=(1,))

    pitch_embeddings = layers.Flatten()(layers.Embedding(128, 64, input_length=1, name="pitch_emb")(pitch_inputs))
    instrument_embeddings = layers.Flatten()(
        layers.Embedding(Config.num_instruments, 8, input_length=1, name="instrument_emb")(instrument_inputs))
    velocity_embeddings = layers.Flatten()(layers.Embedding(128, 4, input_length=1, name="vel_emb")(velocity_inputs))

    x = tf.keras.layers.Concatenate(axis=1)([latent_inputs, pitch_embeddings,
                                             instrument_embeddings, velocity_embeddings])
    # x = layers.Dense(lstm_dim, activation="relu")(x)
    x = layers.RepeatVector(conv_shape[0])(x)
    x = layers.LSTM(lstm_dim, activation="relu", return_sequences=True, dropout=0.1)(x)
    # x = layers.Bidirectional(layers.LSTM(lstm_dim, activation="tanh", return_sequences=True, dropout=0.1))(x)
    # x = layers.TimeDistributed(layers.Dense(conv_shape[1] * units[0], activation="relu"))(x)
    x = layers.Reshape((conv_shape[0], conv_shape[1], int(x.shape[2] / conv_shape[1])))(x)
    for i, (unit, kernel_size, stride) in enumerate(zip(units, kernel_sizes, strides)):
        x = layers.Conv2DTranspose(unit, (kernel_size), activation="relu", strides=(stride), padding="same")(x)
    x = layers.Cropping2D(cropping=((3, 4), (7, 7)))(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="linear", padding="same")(x)
    decoder = keras.Model([latent_inputs, pitch_inputs,
                           instrument_inputs, velocity_inputs],
                          decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


def build_vae(latent_dim, lstm_dim, learning_rate=2e-8, units=(32, 32, 64, 64), kernel_sizes=(3, 3, 3, 3),
              strides=(2, 2, 2, 2)):
    encoder = build_encoder(latent_dim, lstm_dim, units, kernel_sizes, strides)
    decoder = build_decoder(latent_dim, lstm_dim, units, kernel_sizes, strides)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return vae
