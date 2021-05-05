import random
import os
import numpy as np
import data_loader

from audio_processing import get_mel_spectrogram
from typing import Dict
from loguru import logger


class DataGenerator:
    def __init__(self, conf:Dict, batch_size: int = 8):
        assert "csv_file_path" in conf
        assert "base_dir" in conf
        self.conf = conf.copy()
        self.batch_size = batch_size
        self.train, self.valid = data_loader.data_loader(conf)
        self.num_train = len(self.train)
        self.num_valid = len(self.valid)

        logger.info("DataGenerator instantiated")

    def generator(self, set_name: str):
        assert set_name in ["train", "valid"]
        while True:
            x_batch = np.zeros((self.batch_size, self.conf.get("n_mels"), self.conf.get("time_steps")))
            y_batch = np.zeros((self.batch_size, self.conf.get("num_classes")))
            indices = np.random.randint(0, eval(f"self.num_{set_name}"), size=(self.batch_size,))
            current_items = list(eval(f"self.{set_name}.items()"))

            for i, index in enumerate(indices):
                key, value = random.choice(current_items)
                file_path = os.path.join(self.conf.get("base_dir"), f"{key}.wav")

                for j, feature in enumerate(self.conf.get("features")):
                    y_batch[i, j] = int(value[feature] < 50 - self.conf.get("threshold"))
                    y_batch[i, j + 1] = int(value[feature] > 50 + self.conf.get("threshold"))

                x_batch[i] = get_mel_spectrogram(file_path, self.conf) * self.conf.get("scale_factor")

            yield x_batch, y_batch
