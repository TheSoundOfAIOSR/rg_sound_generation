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
    kernels = [conf.default_k] * 4

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

    if conf.use_lstm_in_encoder:
        flattened = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(wrapper[f"out_{len(filters)}"])
        hidden = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(conf.lstm_dim, activation="tanh", recurrent_activation="sigmoid",
                                 return_sequences=False, dropout=conf.lstm_dropout,
                                 recurrent_dropout=0, unroll=False, use_bias=True))(flattened)
    else:
        hidden = tf.keras.layers.Flatten()(wrapper[f"out_{len(filters)}"])
    if conf.is_variational:
        z_mean = tf.keras.layers.Dense(conf.latent_dim, name="z_mean")(hidden)
        z_log_variance = tf.keras.layers.Dense(conf.latent_dim, name="z_log_variance")(hidden)
        z = tf.keras.layers.Lambda(sample_from_latent_space)([z_mean, z_log_variance])
        outputs = [z, z_mean, z_log_variance]
    else:
        outputs = tf.keras.layers.Dense(conf.latent_dim, activation="relu")(hidden)
    model = tf.keras.models.Model(
        encoder_input, outputs, name="encoder"
    )
    tf.keras.utils.plot_model(model, to_file="encoder.png", show_shapes=True)
    if conf.print_model_summary:
        print(model.summary())
    return model


def create_strides_encoder(conf: LocalConfig):
    def conv_block(x, f, k, skip=False):
        x = tf.keras.layers.Conv2D(
            f, k, padding=conf.padding, strides=conf.strides)(x)
        if not skip:
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("elu")(x)
        return x

    if conf is None:
        conf = LocalConfig()

    encoder_input = tf.keras.layers.Input(shape=(conf.row_dim, conf.col_dim, 2), name="encoder_input")

    filters = [32] * 2 + [64] * 2
    kernels = [conf.default_k] * 4
    max_blocks = len(filters)

    filters_kernels = iter(zip(filters, kernels))

    wrapper = {}
    wrapper["block_0_input"] = encoder_input

    for i, (f, k) in enumerate(filters_kernels):
        wrapper[f"block_{i}_output"] = conv_block(wrapper[f"block_{i}_input"], f, k)
        if i == 0:
            wrapper[f"block_{i + 1}_input"] = wrapper[f"block_{i}_output"]
        else:
            wrapper[f"block_{i + 1}_input"] = tf.keras.layers.Add()(
                [wrapper[f"block_{i}_output"], wrapper[f"block_{i}_skip"]]
            )
        if i < max_blocks:
            wrapper[f"block_{i + 1}_skip"] = conv_block(wrapper[f"block_{i}_output"], 1, 1, skip=True)

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
    if conf.print_model_summary:
        print(model.summary())
    return model


def create_1d_encoder(conf: LocalConfig):
    def conv1d_model(inputs):
        filters = [256, 32, 32, 32, 64, 128]
        kernels = [128, 16, 16, 16, 16, 16]
        strides = [8, 2, 2, 2, 2, 2]
        # ToDo: Find better values for kernels for both freq

        x = inputs

        for f, k, s in zip(filters, kernels, strides):
            x = tf.keras.layers.Conv1D(f, k, strides=s, padding=conf.padding)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("elu")(x)

        x = tf.keras.layers.Flatten()(x)
        return x

    if conf is None:
        conf = LocalConfig()

    encoder_input = tf.keras.layers.Input(shape=(conf.row_dim, conf.col_dim, 2), name="encoder_input")
    freq = tf.keras.layers.Lambda(lambda x: x[..., 0])(encoder_input)
    mag = tf.keras.layers.Lambda(lambda x: x[..., 1])(encoder_input)

    freq_flattened = conv1d_model(freq)
    mag_flattened = conv1d_model(mag)

    hidden_input = [freq_flattened, mag_flattened]
    hidden = tf.keras.layers.concatenate(hidden_input)

    if conf.is_variational:
        z_mean = tf.keras.layers.Dense(conf.latent_dim, name="z_mean")(hidden)
        z_log_variance = tf.keras.layers.Dense(conf.latent_dim, name="z_log_variance")(hidden)
        z = tf.keras.layers.Lambda(sample_from_latent_space)([z_mean, z_log_variance])
        outputs = [z, z_mean, z_log_variance]
    else:
        outputs = tf.keras.layers.Dense(conf.latent_dim, activation="elu")(hidden)

    model = tf.keras.models.Model(encoder_input, outputs)
    tf.keras.utils.plot_model(model, to_file="encoder.png", show_shapes=True)
    if conf.print_model_summary:
        print(model.summary())
    return model


def decoder_inputs(conf: LocalConfig):
    z_input = tf.keras.layers.Input(shape=(conf.latent_dim,), name="z_input")
    note_number = tf.keras.layers.Input(shape=(conf.num_pitches,), name="note_number")
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities,), name="velocity")
    return z_input, note_number, velocity


def reshape_z(block, z_input, conf: LocalConfig):
    target_shape = (128 * 2**block, 16 * 2**block, conf.latent_dim)
    n_repeats = target_shape[0] * target_shape[1]
    current_z = tf.keras.layers.RepeatVector(n_repeats)(z_input)
    current_z = tf.keras.layers.Reshape(target_shape=target_shape)(current_z)
    return current_z

def embedding_layers(note_number, velocity, conf: LocalConfig):
    pitch_emb = tf.keras.layers.Embedding(conf.num_pitches, conf.pitch_emb_size, input_length=1, name="pitch_emb")(note_number)
    velocity_emb = tf.keras.layers.Embedding(conf.num_velocities, conf.velocity_emb_size, input_length=1, name="vel_emb")(velocity)
    pitch_emb = tf.keras.layers.Flatten()(pitch_emb)
    velocity_emb = tf.keras.layers.Flatten()(velocity_emb)
    return pitch_emb, velocity_emb


def create_decoder(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    z_input, note_number, velocity = decoder_inputs(conf)
    heuristic_measures = tf.keras.layers.Input(shape=(conf.num_measures,), name="measures")

    if conf.use_encoder:
        if conf.use_embeddings:
            pitch_emb, vel_emb = embedding_layers(note_number, velocity)
            inputs_list = [z_input, pitch_emb, vel_emb]
        else:
            inputs_list = [z_input, note_number, velocity]
    else:
        if conf.use_embeddings:
            pitch_emb, vel_emb = embedding_layers(note_number, velocity)
            inputs_list = [pitch_emb, vel_emb]
        else:
            inputs_list = [note_number, velocity]

    if conf.use_heuristics:
        inputs_list += [heuristic_measures]

    if conf.hidden_dim < conf.latent_dim and conf.check_decoder_hidden_dim:
        conf.hidden_dim = max(conf.hidden_dim, conf.latent_dim)
        print("Decoder hidden dimension updated to", conf.hidden_dim)

    inputs = tf.keras.layers.concatenate(inputs_list)

    if not conf.deep_decoder:
        hidden = tf.keras.layers.Dense(conf.hidden_dim, activation="relu")(inputs)
        num_repeats = int((conf.final_conv_units // conf.hidden_dim) / 2)
        repeat = tf.keras.layers.RepeatVector(num_repeats)(hidden)
        conv_input = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(conf.lstm_dim, activation="tanh", recurrent_activation="sigmoid",
                                 return_sequences=True, dropout=conf.lstm_dropout,
                                 recurrent_dropout=0, unroll=False, use_bias=True))(repeat)
        f_repeat = 2
        k_repeat = 4
        final_conv_shape = conf.final_conv_shape
    else:
        if conf.add_z_to_decoder_blocks:
            print("Deeper decoder can not add z to decoder blocks, disabling conf.add_z_to_decoder_blocks")
            conf.add_z_to_decoder_blocks = False
        units = int(conf.final_conv_units / 16)
        conv_input = tf.keras.layers.Dense(units, activation="relu")(inputs)
        f_repeat = 3
        k_repeat = 6
        final_conv_shape = [int(n/4) for n in conf.final_conv_shape[:-1]] + [192]
    reshaped = tf.keras.layers.Reshape(final_conv_shape)(conv_input)

    filters = list(reversed([32] * f_repeat + [64] * f_repeat))
    kernels = list(reversed([conf.default_k] * k_repeat))
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
        if conf.use_encoder and conf.add_z_to_decoder_blocks:
            wrapper[f"up_conv_{i}"] = tf.keras.layers.Conv2D(
                up_conv_channels, 3, padding="same", name=f"decoder_up_conv_{i}"
            )(wrapper[f"up_out_{i}"])
            current_z = reshape_z(i, z_input, conf)
            z_added = tf.keras.layers.Add()([wrapper[f"up_conv_{i}"], current_z])
            wrapper[f"up_in_{i + 1}"] = tf.keras.layers.concatenate([
                wrapper[f"act_{i}"], z_added
            ])
        else:
            wrapper[f"up_out_{i}"] = tf.keras.layers.Conv2D(1, 1, padding="same")(wrapper[f"up_out_{i}"])
            wrapper[f"up_in_{i + 1}"] = tf.keras.layers.Add()([
                wrapper[f"act_{i}"], wrapper[f"up_out_{i}"]
            ])

    reconstructed = tf.keras.layers.Conv2D(
        2, 3, padding=conf.padding, activation="linear", name="decoder_output",
        kernel_initializer=tf.initializers.glorot_uniform())(wrapper[f"up_in_{len(filters)}"])

    model = tf.keras.models.Model(
        inputs_list,
        reconstructed, name="decoder"
    )
    tf.keras.utils.plot_model(model, to_file="decoder.png", show_shapes=True)
    if conf.print_model_summary:
        print(model.summary())
    return model


def create_rnn_decoder(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    note_number = tf.keras.layers.Input(shape=(conf.num_pitches,), name="note_number")
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities,), name="velocity")
    heuristic_measures = tf.keras.layers.Input(shape=(conf.num_measures,), name="measures")
    if conf.use_embeddings:
        pitch_emb, vel_emb = embedding_layers(note_number, velocity)
        inputs_list = [pitch_emb, vel_emb]
    else:
        inputs_list = [note_number, velocity]

    if conf.use_heuristics:
        inputs_list += [heuristic_measures]

    inputs = tf.keras.layers.concatenate(inputs_list)
    hidden = tf.keras.layers.Dense(256, activation="relu",
                                   kernel_initializer=tf.initializers.glorot_uniform())(inputs)
    num_repeats = 1024
    repeat = tf.keras.layers.RepeatVector(num_repeats)(hidden)
    # Note: When using LSTM, only this specific configuration works with CUDA
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
    if conf.print_model_summary:
        print(model.summary())
    return model


def create_vae(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    if conf.use_max_pool:
        encoder = create_encoder(conf)
    else:
        if conf.encoder_type == "2d":
            encoder = create_strides_encoder(conf)
        else:
            encoder = create_1d_encoder(conf)
    if conf.decoder_type == "rnn":
        decoder = create_rnn_decoder(conf)
    else:
        decoder = create_decoder(conf)

    encoder_input = tf.keras.layers.Input(shape=(conf.row_dim, conf.col_dim, 2))
    note_number = tf.keras.layers.Input(shape=(conf.num_pitches,))
    heuristic_measures = tf.keras.layers.Input(shape=(conf.num_measures,), name="measures")
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities,))

    if conf.is_variational:
        z, z_mean, z_log_variance = encoder(encoder_input)
        inputs = [z, note_number, velocity]
        if conf.use_heuristics:
            inputs += [heuristic_measures]
        reconstruction = decoder(inputs)
        outputs = [reconstruction, z_mean, z_log_variance]
    else:
        z = encoder(encoder_input)
        inputs = [z, note_number, velocity]
        if conf.use_heuristics:
            inputs += [heuristic_measures]
        reconstruction = decoder(inputs)
        outputs = reconstruction

    model_input = [encoder_input, note_number, velocity]
    if conf.use_heuristics:
        model_input += [heuristic_measures]

    model = tf.keras.models.Model(
        model_input,
        outputs,
        name="auto_encoder"
    )
    if conf.print_model_summary:
        print(model.summary())
    return model


def mt_encoder_inputs(conf: LocalConfig):
    time_steps = conf.row_dim
    num_features = conf.col_dim

    f0_shifts = tf.keras.layers.Input(shape=(time_steps, 1), name="f0_shifts")
    h_freq_shifts = tf.keras.layers.Input(shape=(time_steps, num_features), name="h_freq_shifts")
    mag_env = tf.keras.layers.Input(shape=(time_steps, 1), name="mag_env")
    h_mag_dist = tf.keras.layers.Input(shape=(time_steps, num_features), name="h_mag_dist")
    return f0_shifts, h_freq_shifts, mag_env, h_mag_dist


def bn_act(inputs, name=None):
    x = tf.keras.layers.BatchNormalization()(inputs)
    return tf.keras.layers.Activation("elu", name=name)(x)


def conv_1d_encoder_block(inputs, filters, kernel, stride=1, use_act=True, name=None):
    if use_act:
        x = tf.keras.layers.Conv1D(filters, kernel, padding="same", strides=stride)(inputs)
        return bn_act(x, name=name)
    return tf.keras.layers.Conv1D(filters, kernel, padding="same",
                                  strides=stride, name=name)(inputs)


def conv_2d_encoder_block(inputs, filters, kernel, stride=1, use_act=True, name=None):
    if use_act:
        x = tf.keras.layers.Conv2D(filters, kernel, padding="same", strides=stride)(inputs)
        return bn_act(x, name)
    return tf.keras.layers.Conv2D(filters, kernel, padding="same",
                                  strides=stride, name=name)(inputs)


def ffn_block(inputs, num_repeats, conf: LocalConfig):
    x = tf.keras.layers.Dense(conf.hidden_dim, activation="elu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)

    for i in range(0, num_repeats):
        y = tf.keras.layers.Dense(conf.hidden_dim, activation="elu")(x)
        x = tf.keras.layers.Add()([x, y])
        x = tf.keras.layers.LayerNormalization()(x)
    return x


def create_mt_encoder(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()

    f0_shifts, h_freq_shifts, mag_env, h_mag_dist = mt_encoder_inputs(conf)

    f0_shifts_out = conv_1d_encoder_block(f0_shifts, 32, 5)
    h_freq_shifts_out = conv_1d_encoder_block(h_freq_shifts, 64, 5)
    mag_env_out = conv_1d_encoder_block(mag_env, 32, 5)
    h_mag_dist_out = conv_1d_encoder_block(h_mag_dist, 64, 5)

    concat = tf.keras.layers.concatenate(
        [f0_shifts_out, h_freq_shifts_out, mag_env_out, h_mag_dist_out])
    dense = tf.keras.layers.Dense(192, activation="elu")(concat)
    dense = tf.keras.layers.LayerNormalization()(dense)
    block_0 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(dense)

    wrapper = {"block_0": block_0}
    filters = [32, 32, 64, 64, 128, 128]

    for i in range(0, 6):
        if i % 2 == 0:
            wrapper[f"block_{i + 1}"] = conv_2d_encoder_block(wrapper[f"block_{i}"], filters[i], 5, stride=2)
        else:
            conv_out = conv_2d_encoder_block(wrapper[f"block_{i}"], filters[i], 5, stride=2)
            wrapper[f"skip_{i + 1}"] = conv_2d_encoder_block(wrapper[f"block_{i - 1}"],
                                                             1, 1, stride=4, use_act=False)
            wrapper[f"block_{i + 1}"] = tf.keras.layers.Add()([conv_out, wrapper[f"skip_{i + 1}"]])

    hidden = tf.keras.layers.Flatten()(wrapper["block_6"])

    if conf.mt_model_ffn_in_encoder:
        hidden = ffn_block(hidden, 2, conf)

    z = tf.keras.layers.Dense(conf.latent_dim, activation="relu", name="z_output")(hidden)

    m = tf.keras.models.Model(
        [f0_shifts, h_freq_shifts, mag_env, h_mag_dist],
        z
    )
    tf.keras.utils.plot_model(m, to_file="encoder.png", show_shapes=True)
    return m


def create_mt_decoder(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    z_input, note_number, velocity = decoder_inputs(conf)
    if conf.use_embeddings:
        pitch_emb, vel_emb = embedding_layers(note_number, velocity)
        inputs_list = [pitch_emb, vel_emb]
    else:
        inputs_list = [note_number, velocity]

    if conf.use_encoder:
        inputs_list = [z_input] + inputs_list

    if conf.use_heuristics:
        heuristic_measures = tf.keras.layers.Input(shape=(conf.num_measures,), name="measures")
        inputs_list += [heuristic_measures]

    inputs = tf.keras.layers.concatenate(inputs_list)
    ffn_out = ffn_block(inputs, 2, conf)
    dense = tf.keras.layers.Dense(6144, activation="elu")(ffn_out)
    reshaped = tf.keras.layers.Reshape((16, 3, 128))(dense)

    filters = list(reversed([32] * 3 + [64] * 3))
    kernels = [5] * 6
    filters_kernels = iter(zip(filters, kernels))
    wrapper = {
        "up_in_0": reshaped
    }

    for i in range(0, len(filters)):
        f, k = next(filters_kernels)
        wrapper[f"up_out_{i}"] = tf.keras.layers.UpSampling2D(2, name=f"decoder_up_{i}")(wrapper[f"up_in_{i}"])
        wrapper[f"conv_out_{i}"] = tf.keras.layers.Conv2D(
            f, k, padding=conf.padding, name=f"decoder_conv_{i}",
            kernel_initializer=tf.initializers.glorot_uniform())(wrapper[f"up_out_{i}"])
        wrapper[f"bn_out_{i}"] = tf.keras.layers.BatchNormalization(name=f"decoder_bn_{i}")(wrapper[f"conv_out_{i}"])
        wrapper[f"act_{i}"] = tf.keras.layers.Activation("relu", name=f"decoder_act_{i}")(wrapper[f"bn_out_{i}"])
        wrapper[f"up_out_{i}"] = tf.keras.layers.Conv2D(1, 1, padding="same")(wrapper[f"up_out_{i}"])
        wrapper[f"up_in_{i + 1}"] = tf.keras.layers.Add()([
            wrapper[f"act_{i}"], wrapper[f"up_out_{i}"]
        ])

    final_conv_out = wrapper[f"up_in_{len(filters)}"]
    final_conv_out = tf.keras.layers.Conv2D(1, 5, padding=conf.padding, activation="elu")(final_conv_out)

    model_outputs = []

    for k, v in conf.mt_outputs:
        if v["enabled"]:
            units = abs(v["indices"][1] - v["indices"][0])
            split = tf.keras.layers.Lambda(lambda y: y[..., v["indices"][0]:v["indices"][1], :])(final_conv_out)
            squeezed = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1), name=f"{k}_squeeze")(split)
            fc = tf.keras.layers.Dense(units, activation="elu", name=f"{k}_fc")(squeezed)
            conv_1 = conv_1d_encoder_block(fc, 32, 5)
            conv_2 = conv_1d_encoder_block(conv_1, v["channels"], 5, use_act=False, name=f"{k}_out")
            model_outputs.append(conv_2)

    m = tf.keras.models.Model(
        inputs_list, model_outputs
    )
    tf.keras.utils.plot_model(m, to_file="decoder.png", show_shapes=True)
    return m


def create_mt_vae(conf: LocalConfig):
    conf = LocalConfig() if conf is None else conf
    if conf.hidden_dim < conf.latent_dim and conf.check_decoder_hidden_dim:
        conf.hidden_dim = max(conf.hidden_dim, conf.latent_dim)
        print("Changing hidden dim to match latent dim")

    note_number = tf.keras.layers.Input(shape=(conf.num_pitches, ), name="note_number")
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities, ), name="velocity")

    encoder_input = None
    decoder_input = [note_number, velocity]
    encoder_output = None

    if conf.use_encoder:
        f0_shifts, h_freq_shifts, mag_env, h_mag_dist = mt_encoder_inputs(conf)
        encoder_input = [f0_shifts, h_freq_shifts, mag_env, h_mag_dist]
        encoder = create_mt_encoder(conf)
        print("Encoder created")
        encoder_output = encoder(encoder_input)
        decoder_input = [encoder_output] + decoder_input

    if conf.use_heuristics:
        measures = tf.keras.layers.Input(shape=(conf.num_measures, ), name="measures")
        decoder_input += [measures]

    decoder = create_mt_decoder(conf)
    print("Decoder created")
    decoder_output = decoder(decoder_input)

    vae_inputs = encoder_input + decoder_input
    vae_outputs = [encoder_output] + decoder_output

    m = tf.keras.models.Model(
        vae_inputs,
        vae_outputs
    )
    return m


def get_model_from_config(conf):
    if conf.use_encoder:
        print("Creating Auto Encoder")
        return create_vae(conf)
    print("Creating Decoder")
    if conf.decoder_type == "rnn":
        return create_rnn_decoder(conf)
    return create_decoder(conf)
