import os
import pandas as pd
import random

from typing import Dict
from loguru import logger


def data_loader(conf: Dict) -> (Dict, Dict):
    file_path = conf["csv_file_path"]
    base_dir = conf["base_dir"]
    features = conf["features"]
    valid_split = conf["valid_split"]

    assert os.path.isfile(file_path), f"Could not find the file {file_path}"
    assert os.path.isdir(base_dir), f"Could not find the dir {base_dir}"

    def normalize(example):
        count = example["count"]
        if count == 1:
            return example
        updated = {}
        for feature in features:
            updated[feature] = example[feature] / count
        return updated


    logger.info("Loading csv and checking audio files")
    df = pd.read_csv(file_path, index_col=0)

    logger.info("Creating dataset")
    examples = {}

    logger.info(f"Validation split is {valid_split}")

    for i, row in df.iterrows():
        audio_file_name = os.path.splitext(row["audio_file"])[0]
        current_example = dict((feature, row[feature]) for feature in features)
        if audio_file_name in examples:
            for feature in features:
                examples[audio_file_name][feature] += current_example[feature]
            examples[audio_file_name]["count"] += 1
        else:
            examples[audio_file_name] = current_example
            examples[audio_file_name]["count"] = 1

    logger.info("Creating train and valid splits")
    train = {}
    valid = {}

    for key, value in examples.items():
        assert os.path.isfile(os.path.join(base_dir, f"{key}.wav")), f"File not found {key}.wav"
        if random.randint(0, 99) < valid_split * 100:
            valid[key] = normalize(value)
        else:
            train[key] = normalize(value)

    return train, valid
