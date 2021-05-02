import os
import pandas as pd

from typing import Dict, List
from loguru import logger
from tqdm import tqdm


train_files = []
valid_files = []
test_files = []


def create_file_path(file_name: str, base_dir: str) -> (str, str):
    def read_subset(subset):
        return [x for x in os.listdir(os.path.join(base_dir, subset, "audio")) if x.lower().endswith(".wav")]

    def get_path(subset, file_name):
        return os.path.join(base_dir, subset, "audio", file_name)

    global train_files, valid_files, test_files

    if not bool(train_files):
        train_files = read_subset("train")
    if not bool(valid_files):
        valid_files = read_subset("valid")
    if not bool(test_files):
        test_files = read_subset("test")

    if file_name in train_files:
        return get_path("train", file_name), "train"
    if file_name in valid_files:
        return get_path("valid", file_name), "valid"
    if file_name in test_files:
        return get_path("test", file_name), "test"
    raise FileNotFoundError(f"File {file_name} not found")


def data_loader(file_path: str, base_dir: str) -> (Dict, List):
    assert os.path.isfile(file_path), f"Could not find the file {file_path}"
    examples = {
        "train": {},
        "valid": {},
        "test": {}
    }

    logger.info(f"Reading file {file_path}")
    df = pd.read_csv(file_path, index_col=0)
    features = [x for x in df.columns if x not in ["audio_file", "user_id", "description"]]

    logger.info("Populating examples dictionary")
    for i, row in tqdm(df.iterrows()):
        file_name = row["audio_file"]
        file_name = f"{file_name}.wav"
        file_path, set_name = create_file_path(file_name, base_dir)
        examples[set_name][file_name] = {
            "features": dict((feature, row[feature]) for feature in features),
            "file_path": file_path
        }
    return examples, features
