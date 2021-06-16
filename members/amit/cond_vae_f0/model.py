import tensorflow as tf
from config import *
from dataset import num_instruments


def sample_from_latent_space(inputs):
    z_mean, z_log_variance = inputs
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim))
    return z_mean + tf.exp(0.5 * z_log_variance) * epsilon


def create_encoder():
    encoder_input = tf.keras.layers.Input(shape=(input_size,), name="encoder_input")
    hidden = tf.keras.layers.Dense(hidden_units, activation="relu")(encoder_input)
    z_mean = tf.keras.layers.Dense(latent_dim)(hidden)
    z_log_variance = tf.keras.layers.Dense(latent_dim)(hidden)
    z = tf.keras.layers.Lambda(sample_from_latent_space)([z_mean, z_log_variance])
    return tf.keras.models.Model(
        encoder_input, [z, z_mean, z_log_variance], name="encoder"
    )


def create_decoder():
    z_input = tf.keras.layers.Input(shape=(latent_dim, ))
    note_number = tf.keras.layers.Input(shape=(num_pitches, ))
    instrument_id = tf.keras.layers.Input(shape=(num_instruments,))
    inputs = tf.keras.layers.concatenate([z_input, note_number, instrument_id])
    hidden = tf.keras.layers.Dense(hidden_units, activation="relu")(inputs)
    reconstructed = tf.keras.layers.Dense(input_size, activation="relu")(hidden)
    return tf.keras.models.Model(
        [z_input, note_number, instrument_id],
        reconstructed, name="decoder"
    )
