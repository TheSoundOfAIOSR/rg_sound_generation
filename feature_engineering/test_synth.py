import tensorflow as tf
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import tsms

# pip install git+
# https://github.com/fabiodimarco/tf-spectral-modeling-synthesis.git


def exp_envelope(t, alpha):
    alpha = tf.cast(alpha, dtype=tf.float32)
    if tf.size(t) > 0:
        end_time = t[-1]
        if alpha == 0.0:
            envelope = (end_time - t) / end_time
        else:
            exp_t = tf.math.exp(alpha * t / end_time)
            exp_alpha = tf.math.exp(alpha)
            envelope = (exp_t - exp_alpha) / (1.0 - exp_alpha)
            envelope = tf.where(envelope >= 0.0, envelope, 0.0)
    else:
        envelope = t * 0.0

    return envelope


def dahdr_envelope(attack_alpha=0.0, decay_alpha=0.0, release_alpha=0.0,
                   delay_samples=0, attack_samples=0, hold_samples=0,
                   decay_samples=0, release_samples=0):
    t0 = tf.linspace(0.0, 1.0, delay_samples)
    t1 = tf.linspace(0.0, 1.0, attack_samples)
    t2 = tf.linspace(0.0, 1.0, hold_samples)
    t3 = tf.linspace(0.0, 1.0, decay_samples)
    t4 = tf.linspace(0.0, 1.0, release_samples)

    e0 = t0 * 0.0
    e1 = exp_envelope(t1, attack_alpha)
    e2 = t2 * 0.0 + 1.0
    e3 = exp_envelope(t3, decay_alpha)
    e4 = exp_envelope(t4, release_alpha)

    e1 = 1.0 - e1
    e3 = e3[:decay_samples-release_samples]
    if decay_samples - release_samples > 0:
        e4 = e4 * e3[-1]

    return tf.concat([e0, e1, e2, e3, e4], axis=0)


def get_num_harmonics(f0_envelope, sample_rate):
    min_f0 = tf.math.reduce_min(f0_envelope)
    max_period = sample_rate / min_f0
    harmonics = tf.cast(0.5 * max_period, dtype=tf.int32) + 1

    return harmonics


def generate_f0_envelope(note_number, duration, sample_rate, frame_step,
                         vibrato_freq=5.0, vibrato_semitones=1.0):
    frame_rate = sample_rate / frame_step
    frames = int((sample_rate * duration) // frame_step + 1)

    f0 = tsms.core.midi_to_hz(note_number)
    f0_envelope = tf.ones(shape=(1, frames, 1)) * f0
    f0_deviation = 2.0 ** (vibrato_semitones / 12.0)
    f0_inc = f0_deviation * f0 - f0

    t = tf.linspace(0.0, frames / frame_rate, frames)
    t = t[tf.newaxis, :, tf.newaxis]

    f0_variation = f0_inc * tf.math.sin(2.0 * np.pi * vibrato_freq * t)
    f0_envelope = f0_envelope + f0_variation

    return f0_envelope


def generate_h_freq(f0_envelope, sample_rate, frame_step,
                    inharmonicity_amount=0.1,
                    inharmonicity_alpha=2.0):
    harmonics = get_num_harmonics(f0_envelope, sample_rate)
    h_freq = tsms.core.get_harmonic_frequencies(f0_envelope, harmonics)

    inharmonicity_envelope = dahdr_envelope(
        attack_alpha=inharmonicity_alpha,
        attack_samples=harmonics) * inharmonicity_amount

    inharmonicity_envelope = inharmonicity_envelope[tf.newaxis, tf.newaxis, :]

    h_freq *= 1.0 + inharmonicity_envelope

    return h_freq


def generate_h_mag(h_freq, sample_rate, frame_step,
                   mag_alpha_even,
                   mag_alpha_odd,
                   start_alpha, stop_alpha,
                   even_odd_balance=0.5,
                   decay_samples=800,
                   release_samples=200):
    frames = h_freq.shape[1]
    harmonics = h_freq.shape[2]

    attack_alpha = list(np.linspace(0.0, 0.0, harmonics, dtype=np.float))
    decay_alpha = list(np.linspace(start_alpha, stop_alpha, harmonics,
                                   dtype=np.float))
    release_alpha = list(np.linspace(start_alpha, stop_alpha, harmonics,
                                     dtype=np.float))

    t_envelope = []
    for a, d, r in zip(attack_alpha, decay_alpha, release_alpha):
        envelope = dahdr_envelope(
            attack_alpha=float(a),
            decay_alpha=float(d),
            release_alpha=float(r),
            delay_samples=10,
            attack_samples=5,
            hold_samples=0,
            decay_samples=decay_samples,
            release_samples=release_samples)

        envelope = envelope[:frames]
        envelope = tf.pad(envelope, ((0, frames - envelope.shape[0]),))
        envelope = envelope[tf.newaxis, :]
        t_envelope.append(envelope)

    t_envelope = tf.stack(t_envelope, axis=-1)

    mag_even = dahdr_envelope(decay_alpha=mag_alpha_even,
                              release_alpha=mag_alpha_even,
                              decay_samples=harmonics,
                              release_samples=0)
    mag_odd = dahdr_envelope(decay_alpha=mag_alpha_odd,
                             release_alpha=mag_alpha_odd,
                             decay_samples=harmonics,
                             release_samples=0)

    indices = tf.range(0, mag_even.shape[0])
    h_envelope = tf.where(indices % 2 == 0,
                          mag_odd * 2.0 * even_odd_balance,
                          mag_even * 2.0 * (1.0 - even_odd_balance))

    h_envelope = h_envelope[tf.newaxis, tf.newaxis, :]

    h_mag = tf.ones(h_freq.shape) * t_envelope * h_envelope

    return h_mag


def main():
    sample_rate = 16000
    frame_step = 64
    note_number = 50
    duration = 4.0

    # generate f0_envelope
    f0_envelope = generate_f0_envelope(
        note_number, duration, sample_rate, frame_step,
        vibrato_freq=5.0, vibrato_semitones=0.0)

    # generate h_freq
    h_freq = generate_h_freq(f0_envelope, sample_rate, frame_step,
                             inharmonicity_amount=0.05,
                             inharmonicity_alpha=2.0)

    # generate h_mag
    h_mag = generate_h_mag(h_freq, sample_rate, frame_step,
                           mag_alpha_even=-20,
                           mag_alpha_odd=-10,
                           start_alpha=-1, stop_alpha=-30,
                           even_odd_balance=0.8,
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
