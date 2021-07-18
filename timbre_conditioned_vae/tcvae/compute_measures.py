import tensorflow as tf
from .localconfig import LocalConfig
from . import heuristics

heuristic_names = [
    "inharmonicity", "even_odd", "sparse_rich", "attack_rms",
    "decay_rms", "attack_time", "decay_time", "bass", "mid",
    "high_mid", "high"
]


def get_measures(h_freq, h_mag, conf: LocalConfig):
    inharmonic = heuristics.core.inharmonicity_measure(
        h_freq, h_mag, None, None, None, None)
    even_odd = heuristics.core.even_odd_measure(
        h_freq, h_mag, None, None, None, None)
    sparse_rich = heuristics.core.sparse_rich_measure(
        h_freq, h_mag, None, None, None, None)
    attack_rms = heuristics.core.attack_rms_measure(
        h_freq, h_mag, None, None, None, None)
    decay_rms = heuristics.core.decay_rms_measure(
        h_freq, h_mag, None, None, None, None)
    attack_time = heuristics.core.attack_time_measure(
        h_freq, h_mag, None, None, None, None)
    decay_time = heuristics.core.decay_time_measure(
        h_freq, h_mag, None, None, None, None)

    bass = heuristics.core.frequency_band_measure(
            h_freq, h_mag, None, None, conf.sample_rate, None,
            f_min=conf.freq_bands["bass"][0], f_max=conf.freq_bands["bass"][1]
        )
    mid = heuristics.core.frequency_band_measure(
            h_freq, h_mag, None, None, conf.sample_rate, None,
            f_min=conf.freq_bands["mid"][0], f_max=conf.freq_bands["mid"][1]
        )
    high_mid = heuristics.core.frequency_band_measure(
            h_freq, h_mag, None, None, conf.sample_rate, None,
            f_min=conf.freq_bands["high_mid"][0], f_max=conf.freq_bands["high_mid"][1]
        )
    high = heuristics.core.frequency_band_measure(
            h_freq, h_mag, None, None, conf.sample_rate, None,
            f_min=conf.freq_bands["high"][0], f_max=conf.freq_bands["high"][1]
        )

    measures = [inharmonic, even_odd, sparse_rich,
              attack_rms, decay_rms, attack_time, decay_time,
              bass, mid, high_mid, high]
    measures = tf.stack(measures, axis=-1)

    return measures
