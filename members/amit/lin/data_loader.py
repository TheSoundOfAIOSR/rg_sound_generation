import pandas as pd
import numpy as np
import os
from uuid import uuid4
from loguru import logger
from config import Config


def load_data():
    assert os.path.isfile(Config.csv_file_path)
    logger.info("loading csv file")
    df = pd.read_csv(Config.csv_file_path, index_col=0)

    examples = {}
    user_stats = {}
    logger.info("creating examples")
    for i, row in df.iterrows():
        audio_file = row["audio_file"]
        user_id = row["user_id"]
        values = row[1:10].values

        examples[str(uuid4())] = {
            "v": values,
            "audio_file": audio_file,
            "user_id": user_id
        }

        if user_id not in user_stats:
            user_stats[user_id] = [values]
        else:
            user_stats[user_id].append(values)

    mean_std = {}
    for key, value in user_stats.items():
        v = np.array(value).astype("float32")
        mean_std[key] = {
            "mean": np.mean(v, axis=0).tolist(),
            "std": np.std(v, axis=0).tolist()
        }

    eps = 1
    normalized = {}
    logger.info("normalizing examples")
    for key, value in examples.items():
        user_id = value["user_id"]
        normalized[key] = {
            "v": [(value["v"][i] - mean_std[user_id]["mean"][i]) / (eps + mean_std[user_id]["std"][i]) for i in range(0, 9)]
        }
        normalized[key]["audio_file"] = value["audio_file"]
    return normalized
