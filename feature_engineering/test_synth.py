import tensorflow as tf
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import tsms
import core

# pip install git+
# https://github.com/fabiodimarco/tf-spectral-modeling-synthesis.git


def main():
    sample_rate = 16000
    frame_step = 64
    note_number = 50
    duration = 4.0

    # generate f0_envelope
    f0_envelope = core.generate_f0_envelope(
        note_number, duration, sample_rate, frame_step,
        vibrato_freq=5.0, vibrato_semitones=0.0)

    # generate h_freq
    h_freq = core.generate_h_freq(
        f0_envelope, sample_rate, frame_step,
        inharmonicity_amount=0.0,
        inharmonicity_alpha=2.0)

    # generate h_mag
    h_mag = core.generate_h_mag(
        h_freq, sample_rate, frame_step,
        mag_alpha_even=-20,
        mag_alpha_odd=-20,
        start_alpha=-1, stop_alpha=-30,
        even_odd_balance=0.7,
        decay_samples=800,
        release_samples=200)

    # generate h_phase
    initial_h_phase = tf.random.uniform((h_freq.shape[0], 1, h_freq.shape[2]))
    h_phase = tsms.core.generate_phase(h_freq, sample_rate, frame_step,
                                       initial_h_phase=initial_h_phase)

    # generate audio
    audio = tsms.core.harmonic_synthesis(
        h_freq, h_mag, h_phase, sample_rate, frame_step)

    audio = np.squeeze(audio.numpy())
    audio = audio / np.max(np.abs(audio))

    sf.write('samples/synth_audio.wav', audio, sample_rate)

    plt.figure()
    plt.plot(audio)

    def specgrams(x, title):
        plt.figure(figsize=(6.5, 7))
        plt.subplot(2, 1, 1)
        plt.specgram(x, NFFT=256, Fs=sample_rate, window=None,
                     noverlap=256 - frame_step, mode='psd', vmin=-180)
        plt.title(title + ' spectrogram - fft_size = 256')
        plt.subplot(2, 1, 2)
        plt.specgram(x, NFFT=1024, Fs=sample_rate, window=None,
                     noverlap=1024 - frame_step, mode='psd', vmin=-180)
        plt.title(title + ' spectrogram - fft_size = 1024')

    specgrams(audio, title='synth audio')

    plt.show()


if __name__ == '__main__':
    main()
