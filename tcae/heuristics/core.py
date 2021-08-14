import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tsms


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
    t3 = tf.linspace(0.0, 1.0, decay_samples + 1)
    t4 = tf.linspace(0.0, 1.0, release_samples + 1)

    e0 = t0 * 0.0
    e1 = exp_envelope(t1, attack_alpha)
    e2 = t2 * 0.0 + 1.0
    e3 = exp_envelope(t3, decay_alpha)[1:]
    e4 = exp_envelope(t4, release_alpha)[1:]

    e1 = 1.0 - e1
    if decay_samples - release_samples > 0:
        e3 = e3[:decay_samples - release_samples]
        e4 = e4 * e3[-1]
    else:
        e3 = tf.ones(shape=(0,))

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

    h_freq = h_freq * (1.0 + inharmonicity_envelope)

    return h_freq


def generate_h_mag(h_freq, sample_rate, frame_step,
                   mag_alpha_even,
                   mag_alpha_odd,
                   start_alpha, stop_alpha,
                   even_odd_balance=0.5,
                   decay_samples=800,
                   release_samples=200):
    frames = tf.shape(h_freq)[1]
    harmonics = tf.shape(h_freq)[2]

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


def inharmonicity_measure(h_freq, h_mag, h_phase, residual,
                          sample_rate, frame_step):
    h_mag = tf.math.abs(h_mag)

    harmonics = tf.shape(h_freq)[-1]
    harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

    f0 = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)[:, :, tf.newaxis]
    f = h_freq / harmonic_numbers

    h_mag_mean = tf.math.reduce_mean(h_mag, axis=(1, 2), keepdims=True)
    f_variance = tf.math.reduce_mean(
        h_mag * tf.square(f - f0), axis=2, keepdims=True)

    f_variance = tf.where(h_mag_mean > 0.0, f_variance / h_mag_mean, 0.0)

    # remove huge spikes, generally the attack part
    f_variance = tfa.image.median_filter2d(f_variance, filter_shape=(1, 50))
    f_variance = tf.math.reduce_mean(f_variance, axis=(1, 2))

    inharmonic = f_variance ** 2
    inharmonic = inharmonic / (inharmonic + 1.0)

    return inharmonic


def even_odd_measure(h_freq, h_mag, h_phase, residual,
                     sample_rate, frame_step):
    h_mag = tf.math.abs(h_mag)

    even_mean = tf.math.reduce_mean(h_mag[:, :, 1::2], axis=(1, 2))
    odd_mean = tf.math.reduce_mean(h_mag[:, :, 2::2], axis=(1, 2))

    den = even_mean + odd_mean
    even = tf.where(den > 0.0, even_mean / den, 0.0)
    odd = tf.where(den > 0.0, odd_mean / den, 0.0)

    even_odd = 0.5*((1.0 - even) + odd)

    return even_odd


def sparse_rich_measure(h_freq, h_mag, h_phase, residual,
                        sample_rate, frame_step):
    h_mag = tf.math.abs(h_mag)

    db_limit = -60
    h_mag = tsms.core.lin_to_db(h_mag)
    h_mag = h_mag - tf.math.reduce_max(h_mag)
    h_mag = (tf.maximum(h_mag, db_limit) - db_limit) / (-db_limit)

    num_nonzero = tf.math.count_nonzero(
        tf.math.reduce_sum(h_mag, axis=2, keepdims=True),
        axis=1, keepdims=True, dtype=tf.float32)
    num_nonzero = tf.where(num_nonzero == 0.0, 1.0, num_nonzero)

    h_mag_mean = tf.math.reduce_sum(h_mag, axis=1, keepdims=True) / num_nonzero
    h_mag_mean = tf.math.reduce_mean(h_mag_mean, axis=(1, 2))

    sparse_rich = h_mag_mean

    return sparse_rich


def attack_rms_measure(h_freq, h_mag, h_phase, residual,
                       sample_rate, frame_step):
    h_mag = tf.math.abs(h_mag)

    mag = tf.math.reduce_mean(h_mag, axis=2)
    attack_size = tf.math.argmax(mag, axis=1, output_type=tf.int32)
    attack_size = tf.math.minimum(attack_size, tf.shape(mag)[1])

    sample_size = tf.shape(mag)[1]

    def fn(size):
        return tf.concat([
            tf.ones(shape=(size,)),
            tf.zeros(shape=(sample_size - size,))], axis=0)

    attack_mask = tf.map_fn(fn, attack_size, fn_output_signature=tf.float32)

    rms = tf.math.square(attack_mask * mag)
    att_size = tf.math.reduce_sum(attack_mask, axis=1)
    att_size = tf.math.maximum(att_size, 1.0)
    rms = tf.math.reduce_sum(rms, axis=1) / att_size
    rms = tf.math.sqrt(rms + 1e-6)

    return rms


def decay_rms_measure(h_freq, h_mag, h_phase, residual,
                      sample_rate, frame_step):
    h_mag = tf.math.abs(h_mag)

    mag = tf.math.reduce_mean(h_mag, axis=2)
    attack_size = tf.math.argmax(mag, axis=1, output_type=tf.int32)
    attack_size = tf.math.minimum(attack_size, tf.shape(mag)[1])

    sample_size = tf.shape(mag)[1]

    def fn(size):
        return tf.concat([
            tf.ones(shape=(size,)),
            tf.zeros(shape=(sample_size - size,))], axis=0)

    attack_mask = tf.map_fn(fn, attack_size, fn_output_signature=tf.float32)

    db_limit = -80
    mag_db = tsms.core.lin_to_db(mag)
    mag_db = tf.maximum(mag_db, db_limit)
    mag_db = mag_db / (-db_limit) + 1.0
    decay_mask = tf.where(tf.cumsum(mag_db[:, ::-1], axis=1) > 0.0, 1.0, 0.0)
    decay_mask = decay_mask[:, ::-1] - attack_mask

    rms = tf.math.square(decay_mask * mag)
    dec_size = tf.math.reduce_sum(decay_mask, axis=1)
    dec_size = tf.math.maximum(dec_size, 1.0)
    rms = tf.math.reduce_sum(rms, axis=1) / dec_size
    rms = tf.math.sqrt(rms + 1e-6)

    return rms


def attack_time_measure(h_freq, h_mag, h_phase, residual,
                        sample_rate, frame_step):
    h_mag = tf.math.abs(h_mag)

    mag = tf.math.reduce_mean(h_mag, axis=2)
    attack_size = tf.argmax(mag, axis=1, output_type=tf.int32)
    attack_size = tf.cast(attack_size, dtype=tf.float32)

    sample_size = tf.shape(h_mag)[1]
    sample_size = tf.cast(sample_size, dtype=tf.float32)
    sample_size = tf.where(sample_size > 0.0, sample_size, 1.0)
    attack_time = attack_size / sample_size

    return attack_time


def decay_time_measure(h_freq, h_mag, h_phase, residual,
                       sample_rate, frame_step):
    h_mag = tf.math.abs(h_mag)

    mag = tf.math.reduce_mean(h_mag, axis=2)
    attack_size = tf.argmax(mag, axis=1, output_type=tf.int32)

    sample_size = tf.shape(mag)[1]

    def fn(size):
        return tf.concat([
            tf.ones(shape=(size,)),
            tf.zeros(shape=(sample_size - size,))], axis=0)

    attack_mask = tf.map_fn(fn, attack_size, fn_output_signature=tf.float32)

    db_limit = -80
    mag_db = tsms.core.lin_to_db(mag)
    mag_db = tf.maximum(mag_db, db_limit)
    mag_db = mag_db / (-db_limit) + 1.0
    decay_mask = tf.where(tf.cumsum(mag_db[:, ::-1], axis=1) > 0.0, 1.0, 0.0)
    decay_mask = decay_mask[:, ::-1] - attack_mask

    decay_size = tf.math.reduce_sum(decay_mask, axis=1)
    sample_size = tf.cast(sample_size, dtype=tf.float32)
    sample_size = tf.where(sample_size > 0.0, sample_size, 1.0)
    decay_time = decay_size / sample_size

    return decay_time


def frequency_band_measure(h_freq, h_mag, h_phase, residual,
                           sample_rate, frame_step,
                           f_min, f_max):
    h_mag = tf.math.abs(h_mag)

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

    num = tf.math.reduce_sum(h_mag * h * mask, axis=(1, 2))
    den = tf.math.reduce_sum(h_mag * mask, axis=(1, 2))

    ratio = tf.where(den > 0.0, num / den, 0.0)

    return ratio
