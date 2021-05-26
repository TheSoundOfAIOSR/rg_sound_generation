import os
import pandas as pd
from typing import Dict
from loguru import logger


def data_loader(conf: Dict) -> (Dict, Dict):
    file_path = conf["csv_file_path"]
    base_dir = conf["base_dir"]
    features = conf["features"]

    assert os.path.isfile(file_path), f"Could not find the file {file_path}"
    assert os.path.isdir(base_dir), f"Could not find the dir {base_dir}"

    def normalize(example: Dict) -> Dict:
        count = example["count"]
        if count == 1:
            return example
        updated = {}
        for i, feature in enumerate(features):
            updated[feature] = example[feature] / count
        return updated

    def extract_file_name(file_name: str) -> str:
        return file_name.split("+")[-1]

    logger.info("Loading csv and checking audio files")
    df = pd.read_csv(file_path, index_col=0)

    logger.info("Creating dataset")
    examples = {}

    for i, row in df.iterrows():
        audio_file_name = os.path.splitext(row["audio_file"])[0]
        key = audio_file_name
        if not conf.get("pitch_shifted"):
            audio_file_name = extract_file_name(audio_file_name)

        current_example = dict((feature, row[feature]) for feature in features)
        if audio_file_name in examples:
            for feature in features:
                examples[key][feature] += current_example[feature]
            examples[key]["count"] += 1
        else:
            examples[key] = current_example
            examples[key]["count"] = 1
        examples[key]["audio_file_name"] = audio_file_name

    normalized = {}

    for key, value in examples.items():
        assert os.path.isfile(os.path.join(base_dir, f"{key}.wav")), f"File not found {key}.wav"
        normalized[key] = normalize(value)
        normalized[key]["audio_file_name"] = value["audio_file_name"]
    return normalized
