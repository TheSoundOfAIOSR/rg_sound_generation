import tensorflow as tf
import create_model
import data_generator
import config

from typing import Dict, Any
from loguru import logger


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(str(len(gpus)) + "Physical GPUs," + str(len(logical_gpus)) + "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(e)


def train(conf: Dict) -> Any:

    model = create_model.create_model(conf)
    train_gen = data_generator.data_generator(conf, "train", 8)
    valid_gen = data_generator.data_generator(conf, "valid", 8)
    test_gen = data_generator.data_generator(conf, "test", 8)

    logger.info("Starting training")

    _ = model.fit(
        train_gen,
        steps_per_epoch=100,
        validation_data=valid_gen,
        validation_steps=25,
        epochs=40
    )

    logger.info("Training finished")
    _ = model.evaluate(test_gen, steps=10)



if __name__ == "__main__":
    conf = config.get_config()
    train(conf)
