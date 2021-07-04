import tensorflow as tf
from . import tsms


def normalize_h_freq(h_freq, h_mag, note_number):
    f0 = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)
    f0_mean = tf.math.reduce_mean(f0, axis=1)
    note_number = tf.cast(note_number, dtype=tf.float32)
    f0_note = tsms.core.midi_to_hz(note_number)

    harmonics = tf.shape(h_freq)[-1]
    harmonic_indices = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_indices = harmonic_indices[tf.newaxis, tf.newaxis, :]

    st_var = (2.0 ** (1.0 / 12.0) - 1.0)

    h_freq_mean = f0_mean * harmonic_indices
    h_freq_note = f0_note * harmonic_indices

    h_freq_norm = (h_freq - h_freq_mean) / (h_freq_note * st_var)

    return h_freq_norm


def denormalize_h_freq(h_freq_norm, note_number):
    note_number = tf.cast(note_number, dtype=tf.float32)
    f0_note = tsms.core.midi_to_hz(note_number)

    harmonics = tf.shape(h_freq_norm)[-1]
    harmonic_indices = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_indices = harmonic_indices[tf.newaxis, tf.newaxis, :]

    st_var = (2.0 ** (1.0 / 12.0) - 1.0)

    h_freq_note = f0_note * harmonic_indices

    h_freq = h_freq_note * (h_freq_norm * st_var + 1.0)

    return h_freq


def normalize_h_mag(h_mag, db_limit=-120.0):
    h_mag = tsms.core.lin_to_db(h_mag)
    h_mag = h_mag - tf.math.reduce_max(h_mag)
    h_mag_norm = (tf.maximum(h_mag, db_limit) - db_limit) / (-db_limit)

    return h_mag_norm


def denormalize_h_mag(h_mag_norm, db_limit=-120.0):
    h_mag = h_mag_norm * (-db_limit) + db_limit
    h_mag = tsms.core.db_to_lin(h_mag)

    return h_mag
