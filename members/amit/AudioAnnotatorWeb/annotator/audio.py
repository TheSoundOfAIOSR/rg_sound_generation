import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


plt.style.use('dark_background')


def _get_image_file_path(key):
    file_name = f'{key}.png'
    file_path = os.path.join('static', 'images', file_name)
    return file_path


def get_spectrogram(audio_path, key):
    file_path = _get_image_file_path(key)

    if os.path.isfile(file_path):
        return file_path

    plt.clf()
    plt.figure(figsize=(4, 3))

    audio, sr = librosa.load(os.path.join('annotator', audio_path))
    mel = librosa.feature.melspectrogram(
        audio,
        sr=sr,
        n_fft=1024,
        hop_length=64,
        n_mels=256
    )
    log_mel = librosa.power_to_db(mel)
    librosa.display.specshow(
        log_mel,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        cmap='inferno'
    )
    plt.tight_layout()
    plt.savefig(os.path.join('annotator', file_path))
    return file_path
