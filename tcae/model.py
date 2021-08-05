import tensorflow as tf
import json
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


def embedding_layers(inputs, input_dims, embedding_size, conf: LocalConfig):
    numeric_val = tf.keras.layers.Lambda(lambda x: tf.argmax(x, axis=-1))(inputs)
    numeric_val = tf.keras.layers.Lambda(lambda x: x / input_dims)(numeric_val)
    if not conf.scalar_embedding:
        return tf.keras.layers.Embedding(
            input_dims, embedding_size, input_length=1
        )(numeric_val)

    numeric_val = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(numeric_val)
    return numeric_val


def create_decoder(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    z_input, note_number, velocity = decoder_inputs(conf)
    heuristic_measures = tf.keras.layers.Input(shape=(conf.num_measures,), name="measures")

    if conf.use_encoder:
        inputs_list = [z_input, note_number, velocity]
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
    # tf.keras.utils.plot_model(model, to_file="decoder.png", show_shapes=True)
    if conf.print_model_summary:
        print(model.summary())
    return model


def create_rnn_decoder(conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()
    note_number = tf.keras.layers.Input(shape=(conf.num_pitches,), name="note_number")
    velocity = tf.keras.layers.Input(shape=(conf.num_velocities,), name="velocity")
    heuristic_measures = tf.keras.layers.Input(shape=(conf.num_measures,), name="measures")
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


def ffn_block(inputs, num_repeats, hidden_dim, save_names=False, k=None):
    if not save_names:
        x = tf.keras.layers.Dense(hidden_dim, activation="elu")(inputs)
        x = tf.keras.layers.LayerNormalization()(x)

        for i in range(0, num_repeats):
            y = tf.keras.layers.Dense(hidden_dim, activation="elu")(x)
            x = tf.keras.layers.Add()([x, y])
            x = tf.keras.layers.LayerNormalization()(x)
        return x
    else:
        assert k is not None, "Need a key name to save output layer names"
        x = name(k, tf.keras.layers.Dense(hidden_dim, activation="elu"))(inputs)
        x = name(k, tf.keras.layers.LayerNormalization())(x)

        for i in range(0, num_repeats):
            y = name(k, tf.keras.layers.Dense(hidden_dim, activation="elu"))(x)
            x = name(k, tf.keras.layers.Add())([x, y])
            x = name(k, tf.keras.layers.LayerNormalization())(x)
        return x


def get_encoder_inputs(inputs, conf: LocalConfig):
    concat_inputs = []
    for k, v in conf.mt_inputs.items():
        if k in inputs:
            steps = v["shape"][0]
            filters = v["shape"][1]

            padding = steps - inputs[k].shape[1]
            padded = tf.keras.layers.ZeroPadding1D((0, padding))(inputs[k])
            out = conv_1d_encoder_block(padded, filters, 5)
            concat_inputs += [out]
    return concat_inputs


def create_mt_encoder(inputs, conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()

    concat_inputs = get_encoder_inputs(inputs, conf)
    concat = tf.keras.layers.concatenate(concat_inputs)
    dense = tf.keras.layers.Dense(192, activation="elu")(concat)
    dense = tf.keras.layers.LayerNormalization()(dense)
    block_0 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(dense)

    wrapper = {"block_0": block_0}
    filters = [32, 32, 64, 64, 128, 128] if not conf.simple_encoder \
        else [32]*3 + [64]*3

    for i in range(0, 6):
        if i % 2 == 0:
            wrapper[f"block_{i + 1}"] = conv_2d_encoder_block(wrapper[f"block_{i}"], filters[i], 5, stride=2)
        else:
            conv_out = conv_2d_encoder_block(wrapper[f"block_{i}"], filters[i], 5, stride=2)
            wrapper[f"skip_{i + 1}"] = conv_2d_encoder_block(wrapper[f"block_{i - 1}"],
                                                             filters[i], 1, stride=4, use_act=False)
            wrapper[f"block_{i + 1}"] = tf.keras.layers.Add()([conv_out, wrapper[f"skip_{i + 1}"]])

    if conf.simple_encoder:
        flattened = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(wrapper["block_6"])
        hidden = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(conf.lstm_dim, activation="tanh", recurrent_activation="sigmoid",
                                 return_sequences=False, dropout=conf.lstm_dropout,
                                 recurrent_dropout=0, unroll=False, use_bias=True))(flattened)
    else:
        hidden = tf.keras.layers.Flatten()(wrapper["block_6"])
        hidden = ffn_block(hidden, 2, conf.hidden_dim)

    z = tf.keras.layers.Dense(conf.latent_dim, activation="sigmoid", name="z_output")(hidden)

    m = tf.keras.models.Model(inputs, z)
    if conf.print_model_summary:
        tf.keras.utils.plot_model(m, to_file="encoder.png", show_shapes=True,
                                  show_layer_names=False)
    return m


output_layers = {}


def name(key, layer):
    global output_layers

    if key not in output_layers:
        output_layers[key] = []

    layer_name = str(layer.name).split("/")[0]
    output_layers[key].append(layer_name)
    return layer


def save_output_layers():
    global output_layers

    with open("output_layers.json", "w") as f:
        json.dump(output_layers, f)


def create_mt_decoder(inputs, conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()

    concat_inputs = []
    for k, v in inputs.items():
        if conf.use_embeddings:
            if k == "note_number":
                v = embedding_layers(v, conf.num_pitches, conf.pitch_emb_size, conf)
            elif k == "velocity":
                v = embedding_layers(v, conf.num_velocities, conf.velocity_emb_size, conf)
        concat_inputs += [v]

    concat_inputs = tf.keras.layers.concatenate(concat_inputs)

    if conf.simple_decoder:
        # num_units = 3072
        num_repeats = 6
        repeat = tf.keras.layers.RepeatVector(num_repeats)(concat_inputs)
        lstm_out = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(conf.lstm_dim, activation="tanh", recurrent_activation="sigmoid",
                                 return_sequences=True, dropout=conf.lstm_dropout,
                                 recurrent_dropout=0, unroll=False, use_bias=True))(repeat)
        conv_input = tf.keras.layers.Reshape((16, 3, 64))(lstm_out)
    else:
        dense = ffn_block(concat_inputs, 2, conf.hidden_dim)
        num_units = 6144
        dense = tf.keras.layers.Dense(num_units, activation="elu")(dense)
        conv_input = tf.keras.layers.Reshape((16, 3, 128))(dense)

    filters = list(reversed([32] * 3 + [64] * 3))
    kernels = [5] * 6
    filters_kernels = iter(zip(filters, kernels))
    wrapper = {
        "up_in_0": conv_input
    }

    for i in range(0, 6):
        f, k = next(filters_kernels)
        wrapper[f"up_out_{i}"] = tf.keras.layers.UpSampling2D(2, name=f"decoder_up_{i}")(wrapper[f"up_in_{i}"])
        wrapper[f"conv_out_{i}"] = tf.keras.layers.Conv2D(
            f, k, padding=conf.padding, name=f"decoder_conv_{i}",
            kernel_initializer=tf.initializers.glorot_uniform())(wrapper[f"up_out_{i}"])
        wrapper[f"bn_out_{i}"] = tf.keras.layers.BatchNormalization(name=f"decoder_bn_{i}")(wrapper[f"conv_out_{i}"])
        wrapper[f"act_{i}"] = tf.keras.layers.Activation("elu", name=f"decoder_act_{i}")(wrapper[f"bn_out_{i}"])
        if conf.simple_decoder:
            wrapper[f"up_out_{i}"] = tf.keras.layers.Conv2D(f, 1, padding="same")(wrapper[f"up_out_{i}"])
        else:
            wrapper[f"up_out_{i}"] = tf.keras.layers.Dense(units=f)(wrapper[f"up_out_{i}"])
        wrapper[f"up_in_{i + 1}"] = tf.keras.layers.Add()([
            wrapper[f"act_{i}"], wrapper[f"up_out_{i}"]
        ])

    shared_out = wrapper[f"up_in_{len(filters)}"]

    outputs = {}

    for k, v in conf.mt_outputs.items():
        if v["enabled"]:
            repeats = 4
            filters = [v["shape"][2]] * repeats
            kernels = [5] * repeats
            filters_kernels = iter(zip(filters, kernels))

            task_in = name(k, tf.keras.layers.Permute(dims=(1, 3, 2)))(shared_out)
            task_in = name(k, tf.keras.layers.Dense(units=v["shape"][1]))(task_in)
            task_in = name(k, tf.keras.layers.Permute(dims=(1, 3, 2)))(task_in)
            task_in = name(k, tf.keras.layers.Dense(units=v["shape"][2]))(task_in)

            wrapper = {f"up_out_0": task_in}

            for i in range(0, len(filters)):
                _f, _k = next(filters_kernels)
                wrapper[f"conv_out_{i}"] = name(k, tf.keras.layers.Conv2D(
                    _f, _k, padding=conf.padding, name=f"{k}_decoder_conv_{i}"))(wrapper[f"up_out_{i}"])
                wrapper[f"bn_out_{i}"] = name(k, tf.keras.layers.BatchNormalization(
                    name=f"{k}_decoder_bn_{i}"))(wrapper[f"conv_out_{i}"])
                wrapper[f"act_{i}"] = name(k, tf.keras.layers.Activation(
                    "elu", name=f"{k}_decoder_act_{i}"))(wrapper[f"bn_out_{i}"])
                # wrapper[f"up_out_{i}"] = tf.keras.layers.Dense(units=f)(wrapper[f"up_out_{i}"])
                wrapper[f"up_out_{i + 1}"] = name(k, tf.keras.layers.Add()([
                    wrapper[f"act_{i}"], wrapper[f"up_out_{i}"]
                ]))

            conv2d_out = wrapper[f"up_out_{len(filters)}"]
            if conf.using_categorical:
                task_out = name(k, tf.keras.layers.Permute(dims=(1, 3, 2)))(conv2d_out)
                task_out = name(k, tf.keras.layers.Dense(v["channels"], activation="elu"))(task_out)
                task_out = name(k, tf.keras.layers.Permute(dims=(1, 3, 2)))(task_out)
                task_out = name(k, tf.keras.layers.Conv2D(256, 1, padding="same"))(task_out)
            else:
                conv2d_out = name(k, tf.keras.layers.Conv2D(1, 1, padding="same"))(conv2d_out)
                task_out = name(k, tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1)))(conv2d_out)
                task_out = ffn_block(task_out, 2, v["shape"][1], save_names=True, k=k)
                task_out = name(k, tf.keras.layers.Dense(units=v["channels"]))(task_out)
            outputs[k] = task_out

    m = tf.keras.models.Model(
        inputs, outputs
    )
    save_output_layers()
    if conf.print_model_summary:
        tf.keras.utils.plot_model(m, to_file="decoder.png", show_shapes=True,
                                  show_layer_names=False)
    return m


class MtVae(tf.keras.Model):
    def __init__(self, conf: LocalConfig):
        super(MtVae, self).__init__()
        self.conf = LocalConfig() if conf is None else conf
        self.encoder = None
        self.decoder = None

    def build(self, input_shape):
        conf = self.conf

        encoder_inputs = {}
        decoder_inputs = {}

        if conf.use_encoder:
            for k, v in conf.mt_inputs.items():
                if k in input_shape:
                    shape = input_shape[k]
                    encoder_inputs[k] = tf.keras.layers.Input(shape=shape[1:])

            self.encoder = create_mt_encoder(encoder_inputs, conf)
            if conf.print_model_summary:
                print(self.encoder.summary())
            decoder_inputs["z"] = tf.keras.layers.Input(
                shape=(conf.latent_dim,), name="z")

        if conf.use_note_number:
            if "note_number" in input_shape:
                num_inputs = conf.num_pitches if conf.use_one_hot_conditioning else 1
                decoder_inputs["note_number"] = tf.keras.layers.Input(
                    shape=(num_inputs,), name="note_number")
        if conf.use_velocity:
            if "velocity" in input_shape:
                num_inputs = conf.num_velocities if conf.use_one_hot_conditioning else 1
                decoder_inputs["velocity"] = tf.keras.layers.Input(
                    shape=(num_inputs,), name="velocity")
        if conf.use_instrument_id:
            if "instrument_id" in input_shape:
                num_inputs = conf.num_instruments if conf.use_one_hot_conditioning else 1
                decoder_inputs["instrument_id"] = tf.keras.layers.Input(
                    shape=(num_inputs,), name="instrument_id")
        if conf.use_heuristics:
            if "measures" in input_shape:
                decoder_inputs["measures"] = tf.keras.layers.Input(
                    shape=(conf.num_measures,), name="measures")

        if conf.create_decoder_function == 'cnn':
            self.decoder = create_mt_decoder(decoder_inputs, conf)
        else:
            self.decoder = conf.create_decoder_function(decoder_inputs, conf)

        if conf.print_model_summary:
            print(self.decoder.summary())

    def call(self, inputs, training=None, mask=None):
        decoder_inputs = inputs.copy()
        if self.encoder is not None:
            encoder_outputs = self.encoder(inputs)
            decoder_inputs["z"] = encoder_outputs

        decoder_outputs = self.decoder(decoder_inputs)

        return decoder_outputs

    def get_config(self):
        pass


def get_model_from_config(conf):
    print("Creating Auto Encoder")
    mt_vae = MtVae(conf)
    return mt_vae
    # print("Creating Decoder")
    # if conf.decoder_type == "rnn":
    #     return create_rnn_decoder(conf)
    # return create_decoder(conf)
