import librosa
import numpy as np

from typing import Dict


def get_mel_spectrogram(file_path: str, params: Dict) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=params.get("sample_rate"), mono=True)
    mel_spec = librosa.feature.melspectrogram(
        audio,
        sr=params.get("sample_rate"),
        n_fft=params.get("n_fft"),
        hop_length=params.get("hop_len"),
        n_mels=params.get("n_mels")
    )
    return librosa.power_to_db(mel_spec)
