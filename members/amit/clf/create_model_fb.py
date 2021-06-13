import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from typing import Dict


def create_cnn(conf: Dict, input_dim: tuple) -> tf.keras.Model:
    def conv_block(input_, num_filters):
        x = Conv2D(num_filters, 5, strides=2, padding="same")(input_)
        x = BatchNormalization()(x)
        x = Activation("tanh")(x)
        return MaxPool2D(2)(x)

    input_ = Input(shape=input_dim)
    x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_)
    for i in range(0, conf.get("num_conv_blocks")):
        num_filters = min(2**(5 + i), 512)
        x = conv_block(x, num_filters)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    model = tf.keras.models.Model(input_, x)
    return model


def create_mlp(input_dim: tuple) -> tf.keras.Model:
    input_ = Input(shape=input_dim)

    x = Flatten()(input_)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    model = tf.keras.models.Model(input_, x)
    return model


def create_model(conf: Dict, input_shapes, show_summary: bool = False) -> tf.keras.Model:
    cnn = create_cnn(conf, input_shapes["spec"])
    mlp = create_mlp(input_shapes["hpss"])

    input_spec = Input(shape=input_shapes["spec"], name="spec")
    input_hpss = Input(shape=input_shapes["hpss"], name="hpss")

    cnn_out = cnn(input_spec)
    mlp_out = mlp(input_hpss)

    input_ = tf.keras.layers.concatenate([cnn_out, mlp_out])

    x = Dense(128, activation="relu")(input_)
    x = Dropout(0.4)(x)
    output_ = Dense(conf.get("num_classes"), activation="softmax", name="output")(x)

    model = tf.keras.models.Model([input_spec, input_hpss], output_)

    opt = tf.keras.optimizers.Adam(learning_rate=conf.get("learning_rate"))
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    if show_summary:
        print(model.summary())
    return model
