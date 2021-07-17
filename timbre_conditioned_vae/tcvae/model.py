import tensorflow as tf
from .localconfig import LocalConfig


def sample_from_latent_space(inputs):
    z_mean, z_log_variance = inputs
    batch_size = tf.shape(z_mean)[0]
    epsilon = tf.random.normal(shape=(batch_size, LocalConfig().latent_dim))
    return z_mean + tf.exp(0.5 * z_log_variance) * epsilon


def create_encoder(conf: LocalConfig):
    def conv_block(x, f, k, index):
        x = tf.keras.layers.Conv2D(
            f, k, padding=conf.padding, name=f"encoder_conv_{index}",
            kernel_initializer=tf.initializers.glorot_uniform())(x)
        x = tf.keras.layers.BatchNormalization(name=f"encoder_bn_{index}")(x)
        x = tf.keras.layers.Activation("relu", name=f"encoder_act_{index}")(x)
        return x

    if conf is None:
        conf = LocalConfig()

    wrapper = {}

    encoder_input = tf.keras.layers.Input(shape=(conf.row_dim, conf.col_dim, 2), name="encoder_input")

    filters = [32] * 2 + [64] * 2
    kernels = [3] * 4

    filters_kernels = iter(zip(filters, kernels))

    # conv block 1
    f, k = next(filters_kernels)
    wrapper["block_1"] = conv_block(encoder_input, f, k, 0)
    wrapper["out_1"] = tf.keras.layers.MaxPool2D(2, name="encoder_pool_0")(wrapper["block_1"])
    # remaining conv blocks with skip connections
    for i in range(1, len(filters)):
        f, k = next(filters_kernels)
        wrapper[f"block_{i + 1}"] = conv_block(wrapper[f"out_{i}"], f, k, i)
        wrapper[f"skip_{i + 1}"] = tf.keras.layers.concatenate([wrapper[f"block_{i + 1}"], wrapper[f"out_{i}"]])
        wrapper[f"out_{i + 1}"] = tf.keras.layers.MaxPool2D(2, name=f"encoder_pool_{i}")(wrapper[f"skip_{i + 1}"])

    flattened = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(wrapper[f"out_{len(filters)}"])
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(conf.lstm_dim, activation="tanh", recurrent_activation="sigmoid",
                             return_sequences=False, dropout=conf.lstm_dropout,
                             recurrent_dropout=0, unroll=False, use_bias=True))(flattened)
    if conf.is_variational:
        z_mean = tf.keras.layers.Dense(conf.latent_dim,
                                       kernel_initializer=tf.initializers.glorot_uniform(),
                                       name="z_mean")(lstm)
        z_log_variance = tf.keras.layers.Dense(conf.latent_dim,
                                               kernel_initializer=tf.initializers.glorot_uniform(),
                                               name="z_log_variance")(lstm)
        z = tf.keras.layers.Lambda(sample_from_latent_space)([z_mean, z_log_variance])
        outputs = [z, z_mean, z_log_variance]
    else:
        outputs = tf.keras.layers.Dense(conf.latent_dim, activation="relu")(lstm)
    model = tf.keras.models.Model(
        encoder_input, outputs, name="encoder"
    )
    tf.keras.utils.plot_model(model, to_file="encoder.png", show_shapes=True)
    return model


def create_strides_encoder(conf: LocalConfig):
    def conv_block(x, f, k):
        x = tf.keras.layers.Conv2D(
            f, k, padding=conf.padding, strides=conf.strides)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("elu")(x)
        return x

    if conf is None:
        conf = LocalConfig()

    encoder_input = tf.keras.layers.Input(shape=(conf.row_dim, conf.col_dim, 2), name="encoder_input")

    filters = [32] * 2 + [64] * 2
    kernels = [3] * 4
    max_blocks = len(filters)

    filters_kernels = iter(zip(filters, kernels))

    wrapper = {}
    wrapper["block_0_input"] = encoder_input

    for i, (f, k) in enumerate(filters_kernels):
        wrapper[f"block_{i}_output"] = conv_block(wrapper[f"block_{i}_input"], f, k)
        if i == 0:
            wrapper[f"block_{i + 1}_input"] = wrapper[f"block_{i}_output"]
        else:
            wrapper[f"block_{i + 1}_input"] = tf.keras.layers.concatenate(
                [wrapper[f"block_{i}_output"], wrapper[f"block_{i}_skip"]]
            )
        if i < max_blocks:
            wrapper[f"block_{i + 1}_skip"] = conv_block(wrapper[f"block_{i}_output"], conf.skip_channels, 3)

    if conf.use_lstm_in_encoder:
        flattened = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(wrapper[f"block_{max_blocks}_input"])
        hidden = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(conf.lstm_dim, activation="tanh", recurrent_activation="sigmoid",
                                 return_sequences=False, dropout=conf.lstm_dropout,
                                 recurrent_dropout=0, unroll=False, use_bias=True))(flattened)
    else:
        hidden = tf.keras.layers.Flatten()(wrapper[f"block_{max_blocks}_input"])

    if conf.is_variational:
        z_mean = tf.keras.layers.Dense(conf.latent_dim,
                                       kernel_initializer=tf.initializers.glorot_uniform(),
                                       name="z_mean")(hidden)
        z_log_variance = tf.keras.layers.Dense(conf.latent_dim,
                                               kernel_initializer=tf.initializers.glorot_uniform(),
                                               name="z_log_variance")(hidden)
        z = tf.keras.layers.Lambda(sample_from_latent_space)([z_mean, z_log_variance])
        outputs = [z, z_mean, z_log_variance]
    else:
        outputs = tf.keras.layers.Dense(conf.latent_dim, activation="elu")(hidden)
    model = tf.keras.models.Model(
        encoder_input, outputs, name="encoder"
    )
    tf.keras.utils.plot_model(model, to_file="encoder.png", show_shapes=True)
    return model


def decoder_inputs(conf: LocalConfig):
    z_input = tf.keras.layers.Input(shape=(conf.latent_dim,), name="z")
    note_number = tf.keras.layers.Input(shape=(conf.num_pitches,), name="note_number")
    # instrument_id = tf.keras.layers.Input(shape=(conf.num_instruments,), name="instrument_id")
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities,), name="velocity")
    return z_input, note_number, velocity


def reshape_z(block, z_input, conf: LocalConfig):
    target_shape = (128 * 2**block, 16 * 2**block, conf.latent_dim)
    n_repeats = target_shape[0] * target_shape[1]
    current_z = tf.keras.layers.RepeatVector(n_repeats)(z_input)
    current_z = tf.keras.layers.Reshape(target_shape=target_shape)(current_z)
    return current_z


def create_decoder(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    z_input, note_number, velocity = decoder_inputs(conf)
    heuristic_measures = tf.keras.layers.Input(shape=(conf.num_measures,), name="measures")

    if conf.use_encoder:
        inputs_list = [z_input, note_number, velocity, heuristic_measures]
    else:
        inputs_list = [note_number, velocity, heuristic_measures]
    inputs = tf.keras.layers.concatenate(inputs_list)
    hidden = tf.keras.layers.Dense(conf.hidden_dim, activation="relu",
                                   kernel_initializer=tf.initializers.glorot_uniform())(inputs)
    num_repeats = int((conf.final_conv_units // conf.hidden_dim) / 2)
    repeat = tf.keras.layers.RepeatVector(num_repeats)(hidden)
    lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(conf.lstm_dim, activation="tanh", recurrent_activation="sigmoid",
                             return_sequences=True, dropout=conf.lstm_dropout,
                             recurrent_dropout=0, unroll=False, use_bias=True))(repeat)
    reshaped = tf.keras.layers.Reshape(conf.final_conv_shape)(lstm)
    filters = list(reversed([32] * 2 + [64] * 2))
    kernels = list(reversed([3] * 4))
    filters_kernels = iter(zip(filters, kernels))
    wrapper = {
        "up_in_0": reshaped
    }

    up_conv_channels = conf.latent_dim if conf.add_z_to_decoder_blocks else conf.skip_channels

    for i in range(0, len(filters)):
        f, k = next(filters_kernels)
        wrapper[f"up_out_{i}"] = tf.keras.layers.UpSampling2D(2, name=f"decoder_up_{i}")(wrapper[f"up_in_{i}"])

        wrapper[f"conv_out_{i}"] = tf.keras.layers.Conv2D(
            f, k, padding=conf.padding, name=f"decoder_conv_{i}",
            kernel_initializer=tf.initializers.glorot_uniform())(wrapper[f"up_out_{i}"])
        wrapper[f"bn_out_{i}"] = tf.keras.layers.BatchNormalization(name=f"decoder_bn_{i}")(wrapper[f"conv_out_{i}"])
        wrapper[f"act_{i}"] = tf.keras.layers.Activation("relu", name=f"decoder_act_{i}")(wrapper[f"bn_out_{i}"])
        wrapper[f"up_conv_{i}"] = tf.keras.layers.Conv2D(
            up_conv_channels, 3, padding="same", name=f"decoder_up_conv_{i}"
        )(wrapper[f"up_out_{i}"])
        if conf.use_encoder and conf.add_z_to_decoder_blocks:
            current_z = reshape_z(i, z_input, conf)
            z_added = tf.keras.layers.Add()([wrapper[f"up_conv_{i}"], current_z])
            wrapper[f"up_in_{i + 1}"] = tf.keras.layers.concatenate([
                wrapper[f"act_{i}"], z_added
            ])
        else:
            wrapper[f"up_in_{i + 1}"] = tf.keras.layers.concatenate([
                wrapper[f"act_{i}"], wrapper[f"up_conv_{i}"]
            ])

    reconstructed = tf.keras.layers.Conv2D(
        2, 3, padding=conf.padding, activation="linear", name="decoder_output",
        kernel_initializer=tf.initializers.glorot_uniform())(wrapper[f"up_in_{len(filters)}"])

    model = tf.keras.models.Model(
        inputs_list,
        reconstructed, name="decoder"
    )
    tf.keras.utils.plot_model(model, to_file="decoder.png", show_shapes=True)
    return model


def create_rnn_decoder(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    note_number = tf.keras.layers.Input(shape=(conf.num_pitches,), name="note_number")
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities,), name="velocity")
    heuristic_measures = tf.keras.layers.Input(shape=(conf.num_measures,), name="measures")

    inputs_list = [note_number, velocity, heuristic_measures]
    inputs = tf.keras.layers.concatenate(inputs_list)
    hidden = tf.keras.layers.Dense(256, activation="relu",
                                   kernel_initializer=tf.initializers.glorot_uniform())(inputs)
    num_repeats = 1024
    repeat = tf.keras.layers.RepeatVector(num_repeats)(hidden)
    lstm_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, activation="tanh", recurrent_activation="sigmoid",
                             return_sequences=True, dropout=conf.lstm_dropout,
                             recurrent_dropout=0, unroll=False, use_bias=True))(repeat)
    lstm_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, activation="tanh", recurrent_activation="sigmoid",
                             return_sequences=True, dropout=conf.lstm_dropout,
                             recurrent_dropout=0, unroll=False, use_bias=True))(repeat)

    output_1 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(lstm_1)
    output_2 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(lstm_2)

    output = tf.keras.layers.concatenate([output_1, output_2])
    model = tf.keras.models.Model(
        inputs_list,
        output, name="decoder"
    )
    return model


def create_vae(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    if conf.use_max_pool:
        encoder = create_encoder(conf)
    else:
        encoder = create_strides_encoder(conf)
    if conf.decoder_type == "rnn":
        decoder = create_rnn_decoder(conf)
    else:
        decoder = create_decoder(conf)

    encoder_input = tf.keras.layers.Input(shape=(conf.row_dim, conf.col_dim, 2))
    note_number = tf.keras.layers.Input(shape=(conf.num_pitches,))
    # instrument_id = tf.keras.layers.Input(shape=(conf.num_instruments,))
    heuristic_measures = tf.keras.layers.Input(shape=(conf.num_measures,), name="measures")
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities,))

    if conf.is_variational:
        z, z_mean, z_log_variance = encoder(encoder_input)
        reconstruction = decoder([z, note_number, velocity, heuristic_measures])
        outputs = [reconstruction, z_mean, z_log_variance]
    else:
        z = encoder(encoder_input)
        reconstruction = decoder([z, note_number, velocity, heuristic_measures])
        outputs = reconstruction

    model = tf.keras.models.Model(
        [encoder_input, note_number, velocity, heuristic_measures],
        outputs,
        name="auto_encoder"
    )
    return model


def get_model_from_config(conf):
    if conf.use_encoder:
        print("Creating Auto Encoder")
        return create_vae(conf)
    print("Creating Decoder")
    if conf.decoder_type == "rnn":
        return create_rnn_decoder(conf)
    return create_decoder(conf)
