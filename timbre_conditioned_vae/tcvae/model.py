from typing import Tuple
import tensorflow as tf
from .localconfig import LocalConfig


def sample_from_latent_space(inputs):
    z_mean, z_log_variance = inputs
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, LocalConfig().latent_dim))
    return z_mean + tf.exp(0.5 * z_log_variance) * epsilon


def create_encoder(conf: LocalConfig):
    def conv_block(x, f, k):
        x = tf.keras.layers.Conv2D(f, k, padding=conf.padding,
                                   kernel_initializer=tf.initializers.glorot_uniform())(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x

    wrapper = {}

    encoder_input = tf.keras.layers.Input(shape=(conf.row_dim, conf.col_dim, 2), name="encoder_input")

    filters = [32] * 3 + [64] * 3
    kernels = [3] * 6

    filters_kernels = iter(zip(filters, kernels))

    # conv block 1
    f, k = next(filters_kernels)
    wrapper["block_1"] = conv_block(encoder_input, f, k)
    wrapper["out_1"] = tf.keras.layers.MaxPool2D(2)(wrapper["block_1"])
    # remaining conv blocks with skip connections
    for i in range(1, len(filters)):
        f, k = next(filters_kernels)
        wrapper[f"block_{i + 1}"] = conv_block(wrapper[f"out_{i}"], f, k)
        wrapper[f"skip_{i + 1}"] = tf.keras.layers.concatenate([wrapper[f"block_{i + 1}"], wrapper[f"out_{i}"]])
        wrapper[f"out_{i + 1}"] = tf.keras.layers.MaxPool2D(2)(wrapper[f"skip_{i + 1}"])

    flattened = tf.keras.layers.Flatten()(wrapper[f"out_{len(filters)}"])
    hidden = tf.keras.layers.Dense(conf.hidden_dim, activation="relu",
                                   kernel_initializer=tf.initializers.glorot_uniform())(flattened)
    z_mean = tf.keras.layers.Dense(conf.latent_dim,
                                   kernel_initializer=tf.initializers.glorot_uniform())(hidden)
    z_log_variance = tf.keras.layers.Dense(conf.latent_dim,
                                           kernel_initializer=tf.initializers.glorot_uniform())(hidden)
    z = tf.keras.layers.Lambda(sample_from_latent_space)([z_mean, z_log_variance])
    model = tf.keras.models.Model(
        encoder_input, [z, z_mean, z_log_variance], name="encoder"
    )
    tf.keras.utils.plot_model(model, to_file="encoder.png", show_shapes=True)
    return model


def decoder_inputs(conf: LocalConfig):
    z_input = tf.keras.layers.Input(shape=(conf.latent_dim,))
    note_number = tf.keras.layers.Input(shape=(conf.num_pitches,))
    instrument_id = tf.keras.layers.Input(shape=(conf.num_instruments,))
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities,))
    return z_input, note_number, instrument_id, velocity


def create_decoder_mlp(conf: LocalConfig):
    z_input, note_number, instrument_id, velocity = decoder_inputs(conf)
    inputs = tf.keras.layers.concatenate([z_input, note_number, velocity, instrument_id])
    hidden = tf.keras.layers.Dense(conf.hidden_dim, activation="relu",
                                   kernel_initializer=tf.initializers.glorot_uniform())(inputs)
    up_input = tf.keras.layers.Dense(conf.final_conv_units, activation="relu",
                                     kernel_initializer=tf.initializers.glorot_uniform())(hidden)
    x = tf.keras.layers.Reshape((128, 72))(up_input)

    h_freq = tf.keras.layers.Dense(1024, activation="linear")(x)
    h_mag = tf.keras.layers.Dense(1024, activation="linear")(x)

    h_freq = tf.keras.layers.Reshape((1024, 128, 1))(h_freq)
    h_mag = tf.keras.layers.Reshape((1024, 128, 1))(h_mag)

    reconstructed = tf.keras.layers.concatenate([h_freq, h_mag])

    model = tf.keras.models.Model(
        [z_input, note_number, velocity, instrument_id],
        reconstructed, name="decoder"
    )
    tf.keras.utils.plot_model(model, to_file="decoder.png", show_shapes=True)
    return model


def reshape_z(block, z, conf: LocalConfig):
    target_shapes = {
        0: (32, 4, 64),
        1: (64, 8, 64),
        2: (128, 16, 64),
        3: (256, 32, 32),
        4: (512, 64, 32),
        5: (1024, 128, 32)
    }

    s: Tuple[int, int, int] = target_shapes[block]
    units = int(s[0] * s[1] * s[2] / conf.latent_dim)

    x = tf.keras.layers.RepeatVector(units)(z)
    return tf.keras.layers.Reshape(target_shape=s)(x)


def create_decoder_conv(conf: LocalConfig):
    z_input, note_number, instrument_id, velocity = decoder_inputs(conf)
    inputs = tf.keras.layers.concatenate([z_input, note_number, velocity, instrument_id])
    hidden = tf.keras.layers.Dense(conf.hidden_dim, activation="relu",
                                   kernel_initializer=tf.initializers.glorot_uniform())(inputs)
    up_input = tf.keras.layers.Dense(conf.final_conv_units, activation="relu",
                                     kernel_initializer=tf.initializers.glorot_uniform())(hidden)
    x = tf.keras.layers.Reshape(conf.final_conv_shape)(up_input)

    filters = reversed([32] * 3 + [64] * 3)
    kernels = reversed([3] * 6)
    block = 0

    for f, k in zip(filters, kernels):
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Conv2D(f, k, padding=conf.padding,
                                   kernel_initializer=tf.initializers.glorot_uniform())(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        if conf.use_decoder_skip:
            current_z = reshape_z(block, z_input, conf)
            x = tf.keras.layers.Add()([x, current_z])
            block += 1

    reconstructed = tf.keras.layers.Conv2D(2, 3, padding=conf.padding, activation="linear",
                                           kernel_initializer=tf.initializers.glorot_uniform())(x)

    model = tf.keras.models.Model(
        [z_input, note_number, velocity, instrument_id],
        reconstructed, name="decoder"
    )
    tf.keras.utils.plot_model(model, to_file="decoder.png", show_shapes=True)
    return model


def create_decoder(conf: LocalConfig):
    return eval(f"create_decoder_{conf.decoder_type}(conf)")


def create_vae(conf: LocalConfig):
    encoder = create_encoder(conf)
    decoder = create_decoder(conf)

    encoder_input = tf.keras.layers.Input(shape=(conf.row_dim, conf.col_dim, 2))
    note_number = tf.keras.layers.Input(shape=(conf.num_pitches,))
    instrument_id = tf.keras.layers.Input(shape=(conf.num_instruments,))
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities,))

    z, z_mean, z_log_variance = encoder(encoder_input)
    reconstruction = decoder([z, note_number, velocity, instrument_id])

    model = tf.keras.models.Model(
        [encoder_input, note_number, instrument_id, velocity],
        [reconstruction, z_mean, z_log_variance],
        name="VAE"
    )
    return model
