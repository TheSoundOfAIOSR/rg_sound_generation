import tensorflow as tf
from .localconfig import LocalConfig
from . import heuristics


def get_measures(h_freq, h_mag, harmonics, conf: LocalConfig):
    results = None

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

        bass = heuristics.core.frequency_band_measure(
                h_freq_orig, h_mag_orig, None, None, conf.sample_rate, None,
                f_min=conf.freq_bands["bass"][0], f_max=conf.freq_bands["bass"][1]
            )
        mid = heuristics.core.frequency_band_measure(
                h_freq_orig, h_mag_orig, None, None, conf.sample_rate, None,
                f_min=conf.freq_bands["mid"][0], f_max=conf.freq_bands["mid"][1]
            )
        high_mid = heuristics.core.frequency_band_measure(
                h_freq_orig, h_mag_orig, None, None, conf.sample_rate, None,
                f_min=conf.freq_bands["high_mid"][0], f_max=conf.freq_bands["high_mid"][1]
            )
        high = heuristics.core.frequency_band_measure(
                h_freq_orig, h_mag_orig, None, None, conf.sample_rate, None,
                f_min=conf.freq_bands["high"][0], f_max=conf.freq_bands["high"][1]
            )

        result = [inharmonic, even_odd, sparse_rich,
                  attack_rms, decay_rms, attack_time, decay_time,
                  bass, mid, high_mid, high]
        result = tf.convert_to_tensor([result])

        if results is None:
            results = result
        else:
            results = tf.concat([results, result], axis=0)
    return results
