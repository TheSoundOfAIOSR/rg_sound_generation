import random
import shutil
import os
import numpy as np
import data_loader

from audio_processing import get_mel_spectrogram
from typing import Dict
from loguru import logger
from tqdm import tqdm
from pprint import pprint


class DataGenerator:
    def __init__(self, conf:Dict, batch_size: int = 8):
        assert "csv_file_path" in conf
        assert "base_dir" in conf
        self.conf = conf.copy()
        self.batch_size = batch_size
        self.examples = data_loader.data_loader(conf)
        self.num_examples = len(self.examples)
        self.train = {0: [], 1: [], 2: []}
        self.valid = {0: [], 1: [], 2: []}
        self.train_counts = {0: 0, 1: 0, 2: 0}
        self.valid_counts = {0: 0, 1: 0, 2: 0}
        self.num_train = 0
        self.num_valid = 0
        self.classes = [0, 1, 2]

        logger.info("DataGenerator instantiated")
        self.preprocess()
        logger.info("Preprocessing complete")

    def preprocess(self):
        logger.info("Preprocessing examples")

        for c in self.classes:
            folder = os.path.join(self.conf.get("preprocess_dir"), str(c))

            if os.path.isdir(folder):
                shutil.rmtree(folder)
            os.mkdir(folder)

        min_level = 50 - self.conf.get("threshold")
        max_level = 50 + self.conf.get("threshold")
        valid_split = int(self.conf.get("valid_split") * 100)

        logger.info(f"Min level {min_level}, Max level {max_level}")

        for key, value in tqdm(self.examples.items()):
            file_path = os.path.join(self.conf.get("base_dir"), f"{key}.wav")
            current_class = 1

            for j, feature in enumerate(self.conf.get("features")):
                current_val = int(value[feature])
                current_class = 1
                if current_val < min_level:
                    current_class = 0
                elif current_val > max_level:
                    current_class = 2

            target_file_path = os.path.join(self.conf.get("preprocess_dir"), str(current_class), f"{key}.npy")

            x = get_mel_spectrogram(file_path, self.conf) * self.conf.get("scale_factor")
            np.save(target_file_path, x)

            if random.randint(0, 99) < valid_split:
                self.valid[current_class].append(target_file_path)
                self.valid_counts[current_class] += 1
            else:
                self.train[current_class].append(target_file_path)
                self.train_counts[current_class] += 1
        self.num_train = sum(list(self.train_counts.values()))
        self.num_valid = sum(list(self.train_counts.values()))

        logger.info("Class counts in training set")
        pprint(self.train_counts)
        logger.info("Class counts in validation set")
        pprint(self.valid_counts)

    def generator(self, set_name: str):
        assert set_name in ["train", "valid"], "Set name must be either train or valid"
        joint_counter = 0

        while True:
            x_batch = np.zeros((self.batch_size, self.conf.get("n_mels"), self.conf.get("time_steps")))
            y_batch = np.zeros((self.batch_size, self.conf.get("num_classes")))
            current_set = eval(f"self.{set_name}")

            for i in range(0, self.batch_size):
                target_class = joint_counter % len(self.classes)
                example_file = random.choice(current_set[target_class])
                example = np.load(example_file)
                x_batch[i] = example
                y_batch[i, target_class] = 1.
                joint_counter += 1

            yield x_batch, y_batch
