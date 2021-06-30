import tensorflow as tf
import numpy as np
import librosa
from .config import config


def get_audio(audio_file_path: str) -> np.ndarray:
    current_sr = librosa.get_samplerate(audio_file_path)
    assert current_sr == config.sample_rate
    audio, _ = librosa.load(audio_file_path, sr=config.sample_rate, mono=True)
    return audio


def get_stft(
        audio, frame_length: int = config.frame_length,
        frame_step: int = config.frame_step,
        fft_length: int = config.fft_length) -> tf.Tensor:
    return tf.signal.stft(
        tf.cast(audio, tf.float32),
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        pad_end=True
    )


def get_spectrogram(audio: np.ndarray, log: bool = False) -> tf.Tensor:
    audio_stft = get_stft(audio)
    audio_spec = tf.abs(audio_stft)
    if log:
        audio_spec = tf.math.log(audio_spec)
    return tf.transpose(audio_spec)
