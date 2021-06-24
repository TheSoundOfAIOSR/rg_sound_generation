import tensorflow as tf
import numpy as np
import tsms


def features_map(features):
    sample_name = features['sample_name']
    instrument_id = features['instrument_id']
    note_number = features['note_number']
    velocity = features['velocity']
    # instrument_source = features['instrument_source']
    # qualities = features['qualities']
    # audio = features['audio']
    # f0_hz = features['f0_hz']
    # f0_confidence = features['f0_confidence']
    # loudness_db = features['loudness_db']
    f0_estimate = features['f0_estimate']
    h_freq = features['h_freq']
    h_mag = features['h_mag']
    h_phase = features['h_phase']

    # f0_estimate = tf.io.parse_tensor(f0_estimate, out_type=tf.string)
    h_freq = tf.io.parse_tensor(h_freq, out_type=tf.string)
    h_mag = tf.io.parse_tensor(h_mag, out_type=tf.string)
    h_phase = tf.io.parse_tensor(h_phase, out_type=tf.string)

    # f0_estimate = tf.io.parse_tensor(f0_estimate, out_type=tf.float32)
    h_freq = tf.io.parse_tensor(h_freq, out_type=tf.float32)
    h_mag = tf.io.parse_tensor(h_mag, out_type=tf.float32)
    h_phase = tf.io.parse_tensor(h_phase, out_type=tf.float32)

    h_freq = tf.expand_dims(h_freq, axis=0)
    h_mag = tf.expand_dims(h_mag, axis=0)
    h_phase = tf.expand_dims(h_phase, axis=0)

    f0 = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)
    f0_mean = tf.math.reduce_mean(f0, axis=1)
    harmonics = tf.shape(h_freq)[-1]
    harmonic_indices = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_indices = harmonic_indices[tf.newaxis, tf.newaxis, :]
    h_freq_centered = h_freq - (f0_mean * harmonic_indices)
    g_phase = tsms.core.generate_phase(h_freq, sample_rate=16000, frame_step=64)
    d_phase = tsms.core.phase_diff(h_phase, g_phase)
    # unwrap d_phase from +/- pi to +/- 2*pi
    d_phase = tsms.core.phase_unwrap(d_phase, axis=1)
    d_phase = (d_phase + 2.0 * np.pi) % (4.0 * np.pi) - 2.0 * np.pi

    h_freq = tf.squeeze(h_freq, axis=0)
    h_mag = tf.squeeze(h_mag, axis=0)
    h_phase = tf.squeeze(h_phase, axis=0)
    h_freq_centered = tf.squeeze(h_freq_centered, axis=0)
    d_phase = tf.squeeze(d_phase, axis=0)

    h_freq_norm = (h_freq_centered - tf.reduce_mean(h_freq_centered)) / tf.math.reduce_std(h_freq_centered)
    h_mag_norm = (h_mag - tf.reduce_mean(h_mag)) / tf.math.reduce_std(h_mag)
    d_phase_norm = (d_phase - tf.reduce_mean(d_phase)) / tf.math.reduce_std(d_phase)

    element_dict = {
        'sample_name': sample_name,
        'instrument_id': instrument_id,
        'note_number': note_number,
        'velocity': velocity,
        'h_freq': h_freq,
        'h_mag': h_mag,
        'h_phase': h_phase,
        'd_phase': d_phase,
        'h_freq_centered': h_freq_centered,
        'h_freq_norm': h_freq_norm,
        'h_mag_norm': h_mag_norm,
        'd_phase_norm': d_phase_norm
    }
    return element_dict
