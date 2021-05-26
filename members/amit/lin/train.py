import tensorflow as tf
import numpy as np
import os
from loguru import logger
import random
import data_loader
import model
from config import Config


def data_generator(dataset, batch_size):
    examples = list(dataset.items())

    while True:
        x_batch = np.zeros((batch_size, 128, 63))
        y_batch = np.zeros((batch_size, 9))

        for i in range(0, batch_size):
            key, value = random.choice(examples)
            example = value["audio_file"]
            label = value["v"]
            example, _ = os.path.splitext(example)
            file_path = os.path.join(Config.data_dir, f"{example}.spec.npy")
            x_batch[i] = np.load(file_path)
            y_batch[i] = np.array(label)
        yield x_batch, y_batch


def split_data(split=25):
    examples = data_loader.load_data()

    train = {}
    valid = {}

    for key, value in examples.items():
        if random.randint(0, 99) < split:
            valid[key] = value
        else:
            train[key] = value
    return train, valid


def train():
    m = model.create_model()
    train, valid = split_data()

    logger.info(f"training set has {len(train)} examples")
    logger.info(f"validation set has {len(valid)} examples")

    _ = m.fit(
        data_generator(train, 4),
        validation_data=data_generator(valid, 4),
        epochs=200,
        steps_per_epoch=400,
        validation_steps=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=6, verbose=True),
            tf.keras.callbacks.ModelCheckpoint(
                "checkpoints/val_loss_{val_loss:.4f}.h5",
                monitor="val_loss", save_best_only=True, save_weights_only=False, verbose=True
            ),
            tf.keras.callbacks.CSVLogger(f"logs/logs.csv", separator=",", append=False)
        ],
        verbose=2
    )


if __name__ == "__main__":
    train()
