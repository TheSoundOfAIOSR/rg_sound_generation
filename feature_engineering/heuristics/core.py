import tensorflow as tf
import numpy as np
from . import tsms
import matplotlib.pyplot as plt


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
            attack_samples=0,
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


def A_weighting(frequencies, min_db=-80.0):
    def log10(x):
        return tsms.core.logb(x, 10.0)

    f_sq = frequencies ** 2.0

    const = tf.constant([12200, 20.6, 107.7, 737.9]) ** 2.0
    weights = 2.0 + 20.0 * (
        log10(const[0])
        + 2 * log10(f_sq)
        - log10(f_sq + const[0])
        - log10(f_sq + const[1])
        - 0.5 * log10(f_sq + const[2])
        - 0.5 * log10(f_sq + const[3])
    )

    return weights if min_db is None else tf.maximum(min_db, weights)


def peak_iir_frequency_response(w, wc, bw, g_db):
    """Compute peak iir filter frequency response

     Args:
        w: angular frequencies on which the frequency response is computed,
           range [0, pi].
        wc: center angular frequency, range [0, pi].
        bw: bandwidth, range [0, pi].
        g_db: dB gain at center frequency.

                b0 + b1*z^-1 + b2*z^-2
        H(z) = ------------------------
                a0 + a1*z^-1 + a2*z^-2
     """

    e_iw1 = tf.math.exp(-tf.complex(0.0, 1.0) * tf.cast(w, dtype=tf.complex64))
    e_iw2 = tf.math.multiply(e_iw1, e_iw1)

    g_half = tsms.core.db_to_lin(0.5 * g_db)
    g = g_half * g_half
    tan_bw = tf.math.tan(0.5 * bw)
    cos_wc = tf.math.cos(wc)

    b0 = g_half + g * tan_bw
    b1 = -2.0 * g_half * cos_wc
    b2 = g_half - g * tan_bw

    a0 = g_half + tan_bw
    a1 = b1
    a2 = g_half - tan_bw

    b0 = tf.cast(b0, dtype=tf.complex64)
    b1 = tf.cast(b1, dtype=tf.complex64)
    b2 = tf.cast(b2, dtype=tf.complex64)

    a0 = tf.cast(a0, dtype=tf.complex64)
    a1 = tf.cast(a1, dtype=tf.complex64)
    a2 = tf.cast(a2, dtype=tf.complex64)

    num = b0 + b1 * e_iw1 + b2 * e_iw2
    den = a0 + a1 * e_iw1 + a2 * e_iw2

    h = num / den

    return h


def harmonicity_measure(h_freq, h_mag, h_phase, residual,
                        sample_rate, frame_step):
    mag_w = h_mag / tf.math.reduce_max(h_mag)

    harmonics = h_freq.shape[-1]
    harmonic_indices = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_indices = harmonic_indices[tf.newaxis, tf.newaxis, :]

    f0 = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)
    f0 = tf.concat([f0, f0[:, -1:, :]], axis=1)
    f = h_freq / harmonic_indices

    f_variance = tf.math.reduce_mean(
        mag_w * tf.square(f - f0), axis=2) / tf.math.reduce_mean(mag_w)

    f_variance = tf.where(f_variance > 4.0 * tf.math.reduce_mean(f_variance),
                          0.0, f_variance)

    f_variance = tf.math.reduce_mean(f_variance)

    harmonic = 1.0 / f_variance
    inhamonic = f_variance

    den = harmonic + inhamonic
    harmonic = harmonic / den
    inhamonic = inhamonic / den

    return harmonic, inhamonic


def even_odd_measure(h_freq, h_mag, h_phase, residual,
                     sample_rate, frame_step):
    db_limit = -80
    mag_w = tsms.core.lin_to_db(h_mag) + A_weighting(h_freq)
    mag_w = mag_w - tf.math.reduce_max(mag_w)
    mag_w = (tf.maximum(mag_w, db_limit) - db_limit) / (-db_limit)

    # mag_w = h_mag / tf.math.reduce_max(h_mag)

    even_mean = tf.math.reduce_mean(mag_w[:, :, 1::2])
    odd_mean = tf.math.reduce_mean(mag_w[:, :, 2::2])

    den = even_mean + odd_mean
    even = even_mean / den
    odd = odd_mean / den

    return even, odd


def sparse_rich_measure(h_freq, h_mag, h_phase, residual,
                             sample_rate, frame_step):
    db_limit = -60
    mag_w = tsms.core.lin_to_db(h_mag) + A_weighting(h_freq)
    mag_w = mag_w - tf.math.reduce_max(mag_w)
    mag_w = (tf.maximum(mag_w, db_limit) - db_limit) / (-db_limit)

    num_nonzero = tf.math.count_nonzero(
        tf.math.reduce_sum(mag_w, axis=2), axis=1, dtype=tf.float32)

    h_mag_mean = tf.math.reduce_sum(mag_w, axis=1) / num_nonzero
    h_mag_mean = tf.math.reduce_mean(h_mag_mean)

    sparse = 1.0 - h_mag_mean
    rich = h_mag_mean

    return sparse, rich


def vibrato_straight_measure(h_freq, h_mag, h_phase, residual,
                             sample_rate, frame_step):
    mag_w = h_mag / tf.math.reduce_max(h_mag)

    harmonics = h_freq.shape[-1]
    harmonic_indices = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_indices = harmonic_indices[tf.newaxis, tf.newaxis, :]

    f = h_freq / harmonic_indices
    f_mean = tf.math.reduce_mean(f, axis=1)

    f_variance = tf.math.reduce_mean(
        mag_w * tf.square(f - f_mean), axis=2) / tf.math.reduce_mean(mag_w)

    f_variance = tf.where(f_variance > 4.0 * tf.math.reduce_mean(f_variance),
                          0.0, f_variance)

    f_variance = tf.math.reduce_mean(f_variance)

    vibrato = f_variance
    straight = 1.0 / f_variance

    den = vibrato + straight
    vibrato = vibrato / den
    straight = straight / den

    return vibrato, straight


def tremolo_steady_measure(h_freq, h_mag, h_phase, residual,
                             sample_rate, frame_step):
    mag_w = h_mag / tf.math.reduce_max(h_mag)

    harmonics = h_freq.shape[-1]
    harmonic_indices = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_indices = harmonic_indices[tf.newaxis, tf.newaxis, :]

    f = h_freq / harmonic_indices
    f_mean = tf.math.reduce_mean(f, axis=1)

    f_variance = tf.math.reduce_mean(
        mag_w * tf.square(f - f_mean), axis=2) / tf.math.reduce_mean(mag_w)

    f_variance = tf.where(f_variance > 4.0 * tf.math.reduce_mean(f_variance),
                          0.0, f_variance)

    f_variance = tf.math.reduce_mean(f_variance)

    vibrato = f_variance
    straight = 1.0 / f_variance

    den = vibrato + straight
    vibrato = vibrato / den
    straight = straight / den

    return vibrato, straight


def hard_soft_attack_measure(h_freq, h_mag, h_phase, residual,
                             sample_rate, frame_step):
    d_h_freq = h_freq[:, 1:, :] - h_freq[:, :-1, :]
    d_h_freq = tf.concat([d_h_freq, d_h_freq[:, -1:, :]], axis=1)

    d_h_mag = h_mag[:, 1:, :] - h_mag[:, :-1, :]
    d_h_mag = tf.concat([d_h_mag, d_h_mag[:, -1:, :]], axis=1)

    # mag_w = h_mag / tf.math.reduce_max(h_mag)

    db_limit = -100
    mag_w = tsms.core.lin_to_db(h_mag) + A_weighting(h_freq)
    mag_w = mag_w - tf.math.reduce_max(mag_w)
    mag_w = (tf.maximum(mag_w, db_limit) - db_limit) / (-db_limit)

    mag_w_mean = tf.math.reduce_mean(mag_w)

    d_h_freq = tf.math.reduce_mean(mag_w * d_h_freq, axis=2) / mag_w_mean
    d_h_mag = tf.math.reduce_mean(mag_w * d_h_mag, axis=2) / mag_w_mean

    # plt.figure()
    # plt.plot(np.squeeze(d_h_freq.numpy()))
    #
    # plt.figure()
    # plt.plot(np.squeeze(d_h_mag.numpy()))
    #
    # plt.show()

    return 0.0, 0.0


def frequency_band_measure(h_freq, h_mag,
                           h_phase, residual,
                           sample_rate, frame_step,
                           f_min, f_max):
    # remove components above nyquist frequency
    mask = tf.where(
        tf.greater_equal(h_freq, sample_rate / 2.0),
        0.0, 1.0)

    h_freq = h_freq * mask
    w = 2.0 * np.pi * h_freq / sample_rate

    w_min = 2.0 * np.pi * f_min / sample_rate
    w_max = 2.0 * np.pi * f_max / sample_rate

    wc = 0.5 * (w_max + w_min)
    bw = w_max - w_min
    gain_db = tsms.core.lin_to_db(2.0)  # ~ 6 dB

    h = peak_iir_frequency_response(w=w, wc=wc, bw=bw, g_db=gain_db)
    h = tf.math.abs(h) - 1.0

    # num_elems = tf.math.reduce_sum(mask)
    # num = tf.math.reduce_sum(h_mag * h * mask) / num_elems
    # den = tf.math.reduce_sum(h_mag * mask) / num_elems

    num = tf.math.reduce_mean(h_mag * h * mask)
    den = tf.math.reduce_mean(h_mag * mask)

    measure = num / den

    return measure
