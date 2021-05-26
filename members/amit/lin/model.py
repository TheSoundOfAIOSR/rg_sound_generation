import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization,
    Activation, MaxPool2D, Flatten,
    Lambda, Dropout, Dense
)
from config import Config


def create_model():
    def conv_block(input_, num_filters):
        x = Conv2D(num_filters, 5, strides=2, padding="same")(input_)
        x = BatchNormalization()(x)
        x = Activation("tanh")(x)
        return MaxPool2D(2)(x)

    _input = tf.keras.Input(shape=Config.input_shape)
    x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(_input)

    for i in range(0, 3):
        num_filters = min(2**(5 + i), 512)
        x = conv_block(x, num_filters)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    _output = Dense(9, activation="linear")(x)
    model = tf.keras.models.Model(_input, _output)

    model.compile(
        loss="mse",
        optimizer="adam"
    )
    return model
