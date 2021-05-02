import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, BatchNormalization
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from loguru import logger
from typing import Dict


def create_model(conf: Dict) -> tf.keras.Model:
    def conv_block(input_, num_filters):
        x = Conv2D(num_filters, 3)(input_)
        x = BatchNormalization()(x)
        x = Activation("tanh")(x)
        return MaxPool2D(2)(x)

    logger.info("Creating the model")
    input_ = Input(shape=(conf.get("n_mels"), conf.get("time_steps")))
    x = Lambda(lambda x: tf.expand_dims(x, axis=-1))(input_)
    for i in range(0, conf.get("num_conv_blocks")):
        num_filters = min(2**(5 + i), 512)
        x = conv_block(x, num_filters)
    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.25)(x)
    output_ = Dense(conf.get("num_classes"), activation='sigmoid')(x)

    model = tf.keras.models.Model(input_, output_)

    logger.info("Compiling the model")
    opt = tf.keras.optimizers.Adam(learning_rate=conf.get("learning_rate"))
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    print(model.summary())
    return model
