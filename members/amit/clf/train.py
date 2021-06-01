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


def train(conf: Dict, dry: bool = False) -> Any:

    batch_size = 6
    data_gen = data_generator.DataGenerator(conf, batch_size)
    print(data_gen.input_shapes)
    model = create_model.create_model(conf, data_gen.input_shapes, False)

    if dry:
        return

    logger.info("Starting training")

    _ = model.fit(
        data_gen.generator("train"),
        steps_per_epoch=int(data_gen.num_train / batch_size),
        validation_data=data_gen.generator("valid"),
        validation_steps=int(data_gen.num_valid / batch_size),
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6, verbose=True),
            tf.keras.callbacks.ModelCheckpoint(
                f"checkpoints/{conf.get('model_name')}" + "_loss_{val_loss:.4f}_acc_{val_accuracy:.2f}.h5",
                monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=True
            ),
            tf.keras.callbacks.CSVLogger(f"logs/{conf.get('model_name')}.csv", separator=",", append=False)
        ],
        verbose=True
    )

    logger.info("Training finished")


if __name__ == "__main__":
    conf = config.get_config()
    dry = conf.get("dry_run")
    all_features = conf.get("all_features")
    for f in all_features:
        conf["features"] = [f]
        conf["model_name"] = f
        logger.info(f"Starting training for output: {f}")
        logger.info("With configuration:")
        logger.info(conf)
        train(conf, dry)
