import random
import shutil
import os
import numpy as np
import data_loader
import audio_processing

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
        self.train = {}
        self.valid = {}
        self.num_train = 0
        self.num_valid = 0
        self.input_shapes = {
            "spec": (),
            "hpss": ()
        }
        logger.info("DataGenerator instantiated")
        self.preprocess()
        logger.info("Preprocessing complete")

    def preprocess(self):
        logger.info("Preprocessing examples")

        folder = os.path.join(self.conf.get("preprocess_dir"))

        if self.conf.get("reset_data"):
            if os.path.isdir(folder):
                shutil.rmtree(folder)

        if not os.path.isdir(folder):
            os.mkdir(folder)

        valid_split = int(self.conf.get("valid_split") * 100)

        for key, value in tqdm(self.examples.items()):
            audio_file_name = value["audio_file_name"]
            file_path = os.path.join(self.conf.get("base_dir"), f"{audio_file_name}.wav")
            target_file_path = os.path.join(self.conf.get("preprocess_dir"), audio_file_name)
            f = self.conf["features"][0]

            if not os.path.isfile(f"{target_file_path}.spec.npy"):
                spec, hpss = audio_processing.get_features(file_path, self.conf)
                self.input_shapes["spec"] = spec.shape
                self.input_shapes["hpss"] = hpss.shape
                np.save(f"{target_file_path}.spec", spec)
                np.save(f"{target_file_path}.hpss", hpss)
            elif len(self.input_shapes["spec"]) == 0:
                spec = np.load(f"{target_file_path}.spec.npy")
                hpss = np.load(f"{target_file_path}.hpss.npy")
                self.input_shapes["spec"] = spec.shape
                self.input_shapes["hpss"] = hpss.shape

            if random.randint(0, 99) < valid_split:
                self.valid[key] = value[f] / 100.
            else:
                self.train[key] = value[f] / 100.
        self.num_train = len(self.train)
        self.num_valid = len(self.valid)

    def generator(self, set_name: str):
        assert set_name in ["train", "valid"], "Set name must be either train or valid"

        current_set = eval(f"self.{set_name}")
        current_items = list(current_set.items())

        while True:
            spec_batch = np.zeros((self.batch_size,) + self.input_shapes["spec"])
            hpss_batch = np.zeros((self.batch_size,) + self.input_shapes["hpss"])
            y_batch = np.zeros((self.batch_size, 1))

            for i in range(0, self.batch_size):
                example_file, value = random.choice(current_items)
                example_file = os.path.join(self.conf.get("preprocess_dir"), example_file)
                example_spec = np.load(f"{example_file}.spec.npy") * self.conf.get("scale_factor")
                example_hpss = np.load(f"{example_file}.hpss.npy") * self.conf.get("scale_factor")
                spec_batch[i] = example_spec
                hpss_batch[i] = example_hpss
                y_batch[i] = value

            yield {"spec": spec_batch, "hpss": hpss_batch}, {"output": y_batch}
