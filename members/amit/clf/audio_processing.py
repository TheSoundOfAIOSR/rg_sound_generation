import librosa
import numpy as np

from typing import Dict


def get_mel_spectrogram(audio: np.ndarray, params: Dict) -> np.ndarray:
    mel_spec = librosa.feature.melspectrogram(
        audio,
        sr=params.get("sample_rate"),
        n_fft=params.get("n_fft"),
        hop_length=params.get("hop_len"),
        n_mels=params.get("n_mels")
    )
    return librosa.power_to_db(mel_spec)


def get_hpr(audio: np.ndarray, params: Dict) -> (np.ndarray, np.ndarray, np.ndarray):
    D = librosa.stft(
        audio,
        n_fft=params.get("n_fft"),
        hop_length=params.get("hop_len")
    )
    H, P = librosa.decompose.hpss(D)
    return H, P, D - (H + P)


def get_features(file_path: str, params: Dict):
    audio, _ = librosa.load(file_path, sr=params.get("sample_rate"), mono=True)
    h, p, r = get_hpr(audio, params)
    h, p, r = np.abs(h).mean(axis=-1), np.abs(p).mean(axis=-1), np.abs(r).mean(axis=-1)
    dim = h.shape[0]
    hpss = np.concatenate([h, p, r], axis=-1)
    hpss = np.reshape(hpss, (dim * 3, 1))
    spec = get_mel_spectrogram(audio, params)
    return spec, hpss
