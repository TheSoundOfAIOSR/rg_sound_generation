import tensorflow as tf
from .localconfig import LocalConfig
from . import heuristics


def get_measures(h_freq, h_mag, harmonics, conf: LocalConfig):
    results = tf.zeros(shape=(conf.batch_size, 8))

    for i in range(0, conf.batch_size):
        h_freq_orig = tf.expand_dims(h_freq[i, :, :harmonics[i], 0], axis=0)
        h_mag_orig = tf.expand_dims(h_mag[i, :, :harmonics[i], 0], axis=0)

        inharmonic = heuristics.core.inharmonicity_measure(
            h_freq_orig, h_mag_orig, None, None, None, None)
        even_odd = heuristics.core.even_odd_measure(
            h_freq_orig, h_mag_orig, None, None, None, None)
        sparse_rich = heuristics.core.sparse_rich_measure(
            h_freq_orig, h_mag_orig, None, None, None, None)
        attack_rms = heuristics.core.attack_rms_measure(
            h_freq_orig, h_mag_orig, None, None, None, None)
        decay_rms = heuristics.core.decay_rms_measure(
            h_freq_orig, h_mag_orig, None, None, None, None)
        attack_time = heuristics.core.attack_time_measure(
            h_freq_orig, h_mag_orig, None, None, None, None)
        decay_time = heuristics.core.decay_time_measure(
            h_freq_orig, h_mag_orig, None, None, None, None)
        f_m = heuristics.core.frequency_band_measure(
            h_freq_orig, h_mag_orig, None, None, conf.sample_rate, None,
            f_min=200, f_max=4000
        )
        results[i, :] = tf.concat([inharmonic, even_odd, sparse_rich,
                                   attack_rms, decay_rms, attack_time, decay_time, f_m])
    return results
