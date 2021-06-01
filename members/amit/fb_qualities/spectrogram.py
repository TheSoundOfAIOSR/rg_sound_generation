import librosa
import tensorflow as tf
import numpy as np
from .config import Config


def get_audio(audio_file_path: str) -> np.ndarray:
    audio, _ = librosa.load(audio_file_path, sr=Config.sample_rate, mono=True)
    return audio


def get_mel_spectrogram(audio_file_path: str, to_db: bool = True) -> np.ndarray:
    audio, _ = librosa.load(audio_file_path, sr=Config.sample_rate, mono=True)
    mel_spec = librosa.feature.melspectrogram(
        audio,
        sr=Config.sample_rate,
        n_fft=Config.n_fft,
        hop_length=Config.hop_len,
        n_mels=Config.n_mels
    )
    if to_db:
        return librosa.power_to_db(mel_spec)
    return mel_spec


def get_stft(
        audio, frame_length: int = Config.n_fft,
        frame_step: int = Config.hop_len,
        fft_length: int = Config.fft_len) -> tf.Tensor:
    return tf.signal.stft(
        tf.cast(audio, tf.float32),
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )


def get_spectrogram(audio: np.ndarray, log: bool = False) -> tf.Tensor:
    audio_stft = get_stft(audio)
    audio_spec = tf.abs(audio_stft)
    if log:
        return tf.math.log(tf.transpose(audio_spec))
    return tf.transpose(audio_spec)
