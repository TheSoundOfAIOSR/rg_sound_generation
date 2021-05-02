import os
import numpy as np
import librosa
import data_loader

from typing import Dict
from loguru import logger


examples, features = None, None


def get_mel_spectrogram(conf: Dict, audio_file_path: str) -> np.ndarray:
    assert os.path.isfile(audio_file_path), f"No file found at {audio_file_path}"
    audio, _ = librosa.load(audio_file_path, sr=conf.get("sample_rate"), mono=True)
    mel_spec = librosa.feature.melspectrogram(
        audio,
        sr=conf.get("sample_rate"),
        n_fft=conf.get("n_fft"),
        hop_length=conf.get("hop_len"),
        n_mels=conf.get("n_mels")
    )
    return librosa.power_to_db(mel_spec)


def data_generator(conf: Dict, set_name: str, batch_size: int = 8) -> (np.ndarray, np.ndarray):
    global examples, features
    if examples is None or features is None:
        examples, features = data_loader.data_loader(conf.get("csv_file_path"), conf.get("base_dir"))
    current_examples = examples.get(set_name)
    file_names = list(current_examples.keys())
    logger.info(f"Found {len(file_names)} files in set {set_name}")

    while True:
        x_batch = np.zeros((batch_size, conf.get("n_mels"), conf.get("time_steps")))
        y_batch = np.zeros((batch_size, conf.get("num_classes")))
        indices = np.random.randint(0, len(file_names), size=(batch_size,))

        for i, index in enumerate(indices):
            file_name = file_names[index]
            example = current_examples[file_name]

            for j, feature in enumerate(features):
                value = example["features"][feature]
                left = bool(value < 50 - conf.get("threshold"))
                right = bool(value > 50 + conf.get("threshold"))
                y_batch[i, j * 2] = left
                y_batch[i, j * 2 + 1] = right

            file_path = example["file_path"]
            x_batch[i] = get_mel_spectrogram(conf, file_path) * conf.get("scale_factor")

        yield x_batch, y_batch
