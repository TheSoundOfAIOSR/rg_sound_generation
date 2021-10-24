import tensorflow as tf
from tcae.localconfig import LocalConfig


def embedding_layers(inputs, input_dims, embedding_size, conf: LocalConfig):
    numeric_val = tf.keras.layers.Lambda(lambda x: tf.argmax(x, axis=-1))(inputs)
    numeric_val = tf.keras.layers.Lambda(lambda x: x / input_dims)(numeric_val)
    if not conf.scalar_embedding:
        return tf.keras.layers.Embedding(
            input_dims, embedding_size, input_length=1
        )(numeric_val)

    numeric_val = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(numeric_val)
    return numeric_val


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


def ffn_block(inputs, num_repeats, hidden_dim):
    x = tf.keras.layers.Dense(hidden_dim, activation="elu")(inputs)
    x = tf.keras.layers.LayerNormalization()(x)

    for i in range(0, num_repeats):
        y = tf.keras.layers.Dense(hidden_dim, activation="elu")(x)
        x = tf.keras.layers.Add()([x, y])
        x = tf.keras.layers.LayerNormalization()(x)
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

    for k, v in conf.data_handler.outputs.items():
        cols = conf.mt_outputs[k]["shape"][1]
        channels = conf.mt_outputs[k]["shape"][2]

        task_in = tf.keras.layers.Permute(dims=(1, 3, 2))(shared_out)
        task_in = tf.keras.layers.Dense(units=cols)(task_in)
        task_in = tf.keras.layers.Permute(dims=(1, 3, 2))(task_in)
        task_in = tf.keras.layers.Dense(units=channels)(task_in)

        repeats = 4
        filters = [channels] * repeats
        kernels = [5] * repeats
        filters_kernels = iter(zip(filters, kernels))

        wrapper = {f"up_out_0": task_in}

        for i in range(0, len(filters)):
            _f, _k = next(filters_kernels)
            wrapper[f"conv_out_{i}"] = tf.keras.layers.Conv2D(
                _f, _k, padding=conf.padding, name=f"{k}_decoder_conv_{i}")(wrapper[f"up_out_{i}"])
            wrapper[f"bn_out_{i}"] = tf.keras.layers.BatchNormalization(
                name=f"{k}_decoder_bn_{i}")(wrapper[f"conv_out_{i}"])
            wrapper[f"act_{i}"] = tf.keras.layers.Activation(
                "elu", name=f"{k}_decoder_act_{i}")(wrapper[f"bn_out_{i}"])
            # wrapper[f"up_out_{i}"] = tf.keras.layers.Dense(units=f)(wrapper[f"up_out_{i}"])
            wrapper[f"up_out_{i + 1}"] = tf.keras.layers.Add()([
                wrapper[f"act_{i}"], wrapper[f"up_out_{i}"]
            ])

        conv2d_out = wrapper[f"up_out_{len(filters)}"]
        if conf.using_categorical:
            task_out = tf.keras.layers.Permute(dims=(1, 3, 2))(conv2d_out)
            task_out = tf.keras.layers.Dense(v["size"], activation="elu")(task_out)
            task_out = tf.keras.layers.Permute(dims=(1, 3, 2))(task_out)
            task_out = tf.keras.layers.Conv2D(256, 1, padding="same")(task_out)
        else:
            conv2d_out = tf.keras.layers.Conv2D(1, 1, padding="same")(conv2d_out)
            task_out = tf.keras.layers.Lambda(lambda y: tf.squeeze(y, axis=-1))(conv2d_out)
            task_out = ffn_block(task_out, 2, cols)
            task_out = tf.keras.layers.Dense(units=v["size"])(task_out)
        outputs[k] = task_out

    m = tf.keras.models.Model(
        inputs, outputs
    )

    if conf.print_model_summary:
        tf.keras.utils.plot_model(m, to_file="decoder.png", show_shapes=True,
                                  show_layer_names=False)
    return m


class Tables(tf.keras.layers.Layer):
    def __init__(self, num_tables, rows, cols, dropout_rate=0.1):
        super(Tables, self).__init__()
        self.num_tables = tf.constant(num_tables, dtype=tf.int32)
        self.rows = rows
        self.cols = cols

        self.tables = self.add_weight(name='tables',
                                      shape=(num_tables, rows, cols, 1))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x, training=None, mask=None):
        x = tf.expand_dims(x, axis=1)
        tables = tf.expand_dims(self.tables, axis=0)
        tables = self.dropout(tables)
        y = tf.linalg.matmul(x, tables)
        batches = tf.shape(y)[0]
        y = tf.reshape(
            y, shape=(batches, self.num_tables * self.rows, self.cols))

        return y


def base_lc_decoder(input_shape, dropout_rate=0.1):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.LocallyConnected1D(16 * 2, 1, 1)(inputs)
    x = tf.keras.layers.Reshape((128, 16, 2))(x)

    def product(z):
        q, k = tf.unstack(z, axis=-1)
        q = tf.expand_dims(q, axis=-1)
        k = tf.expand_dims(k, axis=-2)
        return tf.linalg.matmul(q, k)

    x = tf.keras.layers.Lambda(lambda z: product(z))(x)
    x = tf.keras.layers.Permute(dims=(2, 3, 1))(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    steps = 128
    for i in range(0, 3):
        steps //= 2
        x = tf.keras.layers.UpSampling2D(2)(x)
        y = tf.keras.layers.Conv2D(steps, 4, padding="same")(x)
        y = tf.keras.layers.Activation("relu")(y)
        y = tf.keras.layers.Dropout(rate=dropout_rate)(y)
        x = tf.keras.layers.Conv2D(steps, 1, padding="same")(x)
        x = tf.keras.layers.Add()([x, y])
        # x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Permute(dims=(3, 1, 2))(x)

    outputs = Tables(64, 16, 128, dropout_rate)(x)

    m = tf.keras.models.Model(
        inputs, outputs
    )
    return m


def create_mt_lc_decoder(inputs, conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()

    concat_inputs = []
    for k, v in inputs.items():
        concat_inputs += [v]

    x = tf.keras.layers.concatenate(concat_inputs)
    x = tf.keras.layers.RepeatVector(128)(x)
    input_shape = x.shape[1:]

    x0 = base_lc_decoder(input_shape, conf.lc_dropout_rate)(x)
    x1 = base_lc_decoder(input_shape, conf.lc_dropout_rate)(x)
    x2 = base_lc_decoder(input_shape, conf.lc_dropout_rate)(x)
    x3 = base_lc_decoder(input_shape, conf.lc_dropout_rate)(x)

    x = tf.keras.layers.Concatenate(axis=-1)([x0, x1, x2, x3])
    outputs = {}

    for k, v in conf.data_handler.outputs.items():
        outputs[k] = tf.keras.layers.Dense(v["size"])(x)

    m = tf.keras.models.Model(
        inputs, outputs
    )
    if conf.print_model_summary:
        tf.keras.utils.plot_model(m, to_file="decoder.png",
                                  show_shapes=True, show_layer_names=False)
    return m


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * units
        self.dense1 = tf.keras.layers.Dense(num_hidden, activation=tf.nn.gelu)
        self.dense2 = tf.keras.layers.Dense(units)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False, *args, **kwargs):
        x = self.dense1(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        return x


class FNetBlock(tf.keras.layers.Layer):
    def __init__(self, units, expansion_factor=2, dropout=0.0):
        super().__init__()
        self.ff = FeedForward(units, expansion_factor, dropout)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False, *args, **kwargs):
        residual = x
        x = tf.math.real(tf.signal.fft2d(tf.cast(x, tf.complex64)))
        x = x + residual
        x = self.norm1(x, training=training)

        residual = x
        x = self.ff(x, training=training)
        x = x + residual
        x = self.norm2(x, training=training)
        return x


def create_fnet_encoder(inputs, conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()

    concat_inputs = []
    for k, v in inputs.items():
        concat_inputs += [v]

    concat_inputs = tf.keras.layers.concatenate(concat_inputs)
    x = tf.keras.layers.Dense(256)(concat_inputs)

    skips = []
    for i in range(8):
        x = FNetBlock(256, dropout=0.0)(x)
        if conf.use_fnet_skip_dense:
            s = tf.keras.layers.Dense(256)(x)
            skips.append(s)
        else:
            skips.append(x)

    x = tf.concat(skips, axis=-1)
    x = tf.keras.layers.Lambda(lambda z: tf.math.reduce_mean(z, axis=1))(x)

    outputs = tf.keras.layers.Dense(
        conf.latent_dim, use_bias=False, activation='sigmoid')(x)

    m = tf.keras.models.Model(
        inputs, outputs
    )

    return m


def create_fnet_decoder(inputs, conf: LocalConfig):
    if conf is None:
        conf = LocalConfig()

    concat_inputs = []
    for k, v in inputs.items():
        concat_inputs += [v]

    concat_inputs = tf.keras.layers.concatenate(concat_inputs)
    x = tf.keras.layers.RepeatVector(1000)(concat_inputs)
    x = tf.keras.layers.LocallyConnected1D(64, 1, 1)(x)
    x = tf.keras.layers.Dense(256)(x)

    skips = []
    for i in range(20):
        x = FNetBlock(256, dropout=0.0)(x)
        if conf.use_fnet_skip_dense:
            s = tf.keras.layers.Dense(256)(x)
            skips.append(s)
        else:
            skips.append(x)

    x = tf.concat(skips, axis=-1)

    outputs = {}

    for k, v in conf.data_handler.outputs.items():
        y = tf.keras.layers.Dense(v["size"], use_bias=False)(x)
        outputs[k] = y

    m = tf.keras.models.Model(
        inputs, outputs
    )

    return m


class TCAEModel(tf.keras.Model):
    def __init__(self, conf: LocalConfig):
        super(TCAEModel, self).__init__()
        self.conf = LocalConfig() if conf is None else conf
        self.encoder = None
        self.decoder = None

    def build(self, input_shape):
        conf = self.conf

        encoder_inputs = {}
        decoder_inputs = {}

        if conf.use_encoder:
            for k, v in conf.data_handler.outputs.items():
                if k in input_shape:
                    shape = input_shape[k]
                    encoder_inputs[k] = tf.keras.layers.Input(shape=shape[1:])

            if conf.create_encoder_function == 'mt':
                self.encoder = create_mt_encoder(encoder_inputs, conf)
            elif conf.create_encoder_function == 'fnet':
                self.encoder = create_fnet_encoder(encoder_inputs, conf)
            else:
                self.encoder = conf.create_encoder_function(encoder_inputs, conf)

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
        elif conf.create_decoder_function == 'lc':
            self.decoder = create_mt_lc_decoder(decoder_inputs, conf)
        elif conf.create_decoder_function == 'fnet':
            self.decoder = create_fnet_decoder(decoder_inputs, conf)
        else:
            self.decoder = conf.create_decoder_function(decoder_inputs, conf)

        if conf.print_model_summary:
            print(self.decoder.summary())

    def call(self, inputs, training=None, mask=None):
        decoder_inputs = inputs.copy()
        if self.encoder is not None:
            encoder_outputs = self.encoder(
                inputs, training=training, mask=mask)
            decoder_inputs["z"] = encoder_outputs

        decoder_outputs = self.decoder(
            decoder_inputs, training=training, mask=mask)

        return decoder_outputs

    def get_config(self):
        pass
