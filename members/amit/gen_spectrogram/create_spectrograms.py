import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import click

from tqdm import tqdm


@click.command()
@click.option("--audio_dir")
def create_spectrograms(audio_dir):
    files = [x for x in os.listdir(audio_dir) if x.lower().endswith('.wav')]

    for f in tqdm(files):
        audio_path = os.path.join(audio_dir, f)
        image_path = os.path.join(audio_dir, f'{os.path.splitext(f)[0]}.png')
        audio, sr = librosa.load(audio_path)
        mel = librosa.feature.melspectrogram(
            audio,
            sr=sr,
            n_fft=1024,
            hop_length=64,
            n_mels=256
        )
        log_mel = librosa.power_to_db(mel)

        plt.clf()
        plt.figure(figsize=(4, 3))
        librosa.display.specshow(
            log_mel,
            sr=sr,
            x_axis='time',
            y_axis='mel',
            cmap='inferno'
        )
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()


if __name__ == '__main__':
    create_spectrograms()
