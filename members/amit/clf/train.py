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

    batch_size = 4
    model = create_model.create_model(conf)
    data_gen = data_generator.DataGenerator(conf, batch_size)

    logger.info("Starting training")

    _ = model.fit(
        data_gen.generator("train"),
        steps_per_epoch=int(data_gen.num_train / batch_size),
        validation_data=data_gen.generator("valid"),
        validation_steps=int(data_gen.num_valid / batch_size),
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6),
            tf.keras.callbacks.ModelCheckpoint(
                f"checkpoints/{conf.get('model_name')}" + "_{val_loss:.4f}.h5",
                monitor="val_loss", save_best_only=True, save_weights_only=False
            )
        ],
        verbose=2
    )

    logger.info("Training finished")


if __name__ == "__main__":
    conf = config.get_config()
    all_features = conf.get("all_features")
    for f in all_features:
        conf["features"] = [f]
        conf["model_name"] = f
        logger.info(f"Starting training for output: {f}")
        logger.info("With configuration:")
        logger.info(conf)
        train(conf)
