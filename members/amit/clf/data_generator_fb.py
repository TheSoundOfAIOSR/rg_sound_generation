import json
import random
import os
import numpy as np
import audio_processing
from typing import Dict
from loguru import logger
from tqdm import tqdm


class DataGenerator:
    def __init__(self, conf: Dict, batch_size: int = 8):
        assert "fb_categorical_file" in conf
        assert "base_dir" in conf
        self.conf = conf.copy()
        self.batch_size = batch_size
        with open(conf.get("fb_categorical_file"), "r") as f:
            self.examples = json.load(f)
        self.num_examples = len(self.examples)
        self.train = {}
        self.valid = {}
        self.num_train = 0
        self.num_valid = 0
        fb_qualities = ["fb_thin","fb_hollow","fb_dark","fb_warm","fb_full", "fb_muddy",
                        "fb_bright","fb_boxy","fb_honky","fb_harsh","fb_tinny","fb_sibilance"]
        self.class_to_index = dict((k, i) for i, k in enumerate(fb_qualities))
        self.index_to_class = dict((v, k) for k, v in self.class_to_index.items())
        self.input_shapes = {
            "spec": (128, 63),
            "hpss": (3075, 1)
        }
        logger.info("DataGenerator instantiated")
        self.preprocess()
        logger.info("Preprocessing complete")

    def preprocess(self):
        logger.info("Preprocessing examples")
        valid_split = int(self.conf.get("valid_split") * 100)
        folder = os.path.join(self.conf.get("preprocess_dir"))

        for key, value in tqdm(self.examples.items()):
            if random.randint(0, 99) < valid_split:
                self.valid[key] = value
            else:
                self.train[key] = value

            audio_file_name = os.path.splitext(key)[0]
            file_path = os.path.join(self.conf.get("base_dir"), f"{audio_file_name}.wav")
            target_file_path = os.path.join(self.conf.get("preprocess_dir"), audio_file_name)

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

        self.num_train = len(self.train)
        self.num_valid = len(self.valid)
        logger.info(f"Examples in training set: {self.num_train}")
        logger.info(f"Examples in validation set: {self.num_valid}")

    def generator(self, set_name: str):
        assert set_name in ["train", "valid"], "Set name must be either train or valid"

        while True:
            spec_batch = np.zeros((self.batch_size,) + self.input_shapes["spec"])
            hpss_batch = np.zeros((self.batch_size,) + self.input_shapes["hpss"])
            y_batch = np.zeros((self.batch_size, len(self.class_to_index)))
            current_set = eval(f"self.{set_name}")

            for i in range(0, self.batch_size):
                example_file = random.choice(list(current_set.keys()))
                classes = current_set[example_file]
                example_file = os.path.join(self.conf.get("preprocess_dir"), example_file)
                example_file = os.path.splitext(example_file)[0]
                example_spec = np.load(f"{example_file}.spec.npy") * self.conf.get("scale_factor")
                example_hpss = np.load(f"{example_file}.hpss.npy") * self.conf.get("scale_factor")
                spec_batch[i] = example_spec
                hpss_batch[i] = example_hpss
                for c in classes:
                    y_batch[i, self.class_to_index[c]] = 1.

            yield {"spec": spec_batch, "hpss": hpss_batch}, {"output": y_batch}
