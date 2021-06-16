import tensorflow as tf
from config import *


def sample_from_latent_space(inputs):
    z_mean, z_log_variance = inputs
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, latent_dim))
    return z_mean + tf.exp(0.5 * z_log_variance) * epsilon


def create_encoder():
    encoder_input = tf.keras.layers.Input(shape=(input_size, output_dim), name="encoder_input")

    x = encoder_input

    filters = [32, 32, 64, 64, 128]
    kernels = [7, 7, 5, 5, 3, 3]

    for f, k in zip(filters, kernels):
        x = tf.keras.layers.Conv1D(f, k, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("tanh")(x)
        x = tf.keras.layers.MaxPool1D(2)(x)

    flattened = tf.keras.layers.Flatten()(x)
    hidden = tf.keras.layers.Dense(hidden_units, activation="relu")(flattened)
    z_mean = tf.keras.layers.Dense(latent_dim)(hidden)
    z_log_variance = tf.keras.layers.Dense(latent_dim)(hidden)
    z = tf.keras.layers.Lambda(sample_from_latent_space)([z_mean, z_log_variance])
    return tf.keras.models.Model(
        encoder_input, [z, z_mean, z_log_variance], name="encoder"
    )


def create_decoder():
    z_input = tf.keras.layers.Input(shape=(latent_dim, ))
    note_number = tf.keras.layers.Input(shape=(num_pitches, ))
    instrument_id = tf.keras.layers.Input(shape=(num_classes,))
    inputs = tf.keras.layers.concatenate([z_input, note_number, instrument_id])
    hidden = tf.keras.layers.Dense(hidden_units, activation="relu")(inputs)
    up_input = tf.keras.layers.Dense(4096, activation="relu")(hidden)
    x = tf.keras.layers.Reshape((32, 128))(up_input)

    filters = reversed([32, 32, 64, 64, 128])
    kernels = reversed([7, 7, 5, 5, 3, 3])

    for f, k in zip(filters, kernels):
        x = tf.keras.layers.UpSampling1D(2)(x)
        x = tf.keras.layers.Conv1D(f, k, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("tanh")(x)

    reconstructed = tf.keras.layers.Conv1D(output_dim, 3, padding=padding, activation="tanh")(x)
    return tf.keras.models.Model(
        [z_input, note_number, instrument_id],
        reconstructed, name="decoder"
    )
