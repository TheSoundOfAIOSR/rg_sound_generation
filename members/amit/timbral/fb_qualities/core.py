import os.path
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
from .spectrogram import get_spectrogram
from .config import Config


class FBQuality:
    def __init__(self,
                 name: str,
                 frequencies: List[float],
                 thresholds: List[float],
                 weight: bool = False):
        assert len(frequencies) == 2, f"There must be exactly 2 values in " \
                               f"the list band, found {len(frequencies)}"
        assert len(thresholds) == 2, f"There must be exactly 2 values in " \
                                     f"the list thresholds, found {len(thresholds)}"
        assert frequencies[0] < frequencies[1], "First frequency in the band can not " \
                                  "be higher than second frequency"
        assert thresholds[0] < thresholds[1], "First threshold can not be " \
                                              "higher than second threshold"

        self._name = name
        self._frequencies = frequencies
        self._thresholds = thresholds
        self._weight = weight
        self._band = None
        self._hz_to_band()

    @property
    def name(self):
        return self._name

    @staticmethod
    def time_step_to_angle(time_step: int, num_bands: int) -> float:
        return (time_step / num_bands) * np.pi

    @staticmethod
    def get_arc(num_bands: int) -> np.ndarray:
        arc_range = [FBQuality.time_step_to_angle(i, num_bands) for i in range(0, num_bands)]
        return np.sin(arc_range)

    def _hz_to_band(self) -> None:
        factor = Config.sample_rate / Config.fft_len
        max_ = int(0.5 * Config.sample_rate / factor) + 1
        f1, f2 = self._frequencies
        f1, f2 = int(f1 / factor), min(max_, int(f2 / factor))
        self._band = [f1, f2]

    def get_spectrograms(self, audio: np.ndarray,
                         spec: np.ndarray = None) -> (tf.Tensor, tf.Tensor):
        if spec is None:
            spec = get_spectrogram(audio)
        sliced_spec = spec[self._band[0]: self._band[1], :]
        return sliced_spec, spec

    def get_sliced_fr(self, spec: tf.Tensor) -> tf.Tensor:
        sliced_fr = tf.reduce_sum(spec, axis=-1)
        if self._weight:
            num_bands = len(sliced_fr)
            arc = FBQuality.get_arc(num_bands)
            return sliced_fr * arc
        return sliced_fr

    def get_ratio_and_sliced_fr(self, audio: np.ndarray,
                                spec: np.ndarray = None) -> (float, tf.Tensor):
        sliced_spec, spec = self.get_spectrograms(audio, spec)
        sliced_fr = self.get_sliced_fr(sliced_spec)
        ratio = tf.reduce_sum(sliced_fr) / tf.reduce_sum(spec)
        return ratio.numpy(), sliced_fr

    def get_ratio(self, audio: np.ndarray,
                  spec: np.ndarray = None) -> float:
        ratio, _ = self.get_ratio_and_sliced_fr(audio, spec)
        return ratio

    def is_quality(self, audio: np.ndarray,
                   spec: np.ndarray = None) -> bool:
        ratio = self.get_ratio(audio, spec)
        return self._thresholds[0] <= ratio <= self._thresholds[1]


class FBQualities:
    """
    List of frequency based qualities
    Must be built using the build method before using
    """
    _instance = None
    _names = []
    _is_built = False
    _csv_file_path = ""

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FBQualities, cls).__new__(cls)
        return cls._instance

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def names(self) -> List[str]:
        return self._names

    def _build_from_qualities(self, qualities: List[FBQuality]) -> None:
        for q in qualities:
            vars(self)[q.name] = q
        self._is_built = True

    def build(self, csv_file_path: str = "") -> None:
        self._csv_file_path = csv_file_path
        if csv_file_path == "":
            cwd = os.getcwd()
            csv_file_path = os.path.join(cwd, "list_of_fb_qualities.csv")
        assert os.path.isfile(csv_file_path), f"File does not exist: {csv_file_path}"
        df = pd.read_csv(csv_file_path, index_col=0)
        read_qualities = []
        self._names = []

        for _, row in df.iterrows():
            name = row["name"]
            freq_low = float(row["freq_low"])
            freq_high = float(row["freq_high"])
            frequencies = [freq_low, freq_high]
            thres_low = float(row["thres_low"])
            thres_high = float(row["thres_high"])
            thresholds = [thres_low, thres_high]

            read_qualities.append(FBQuality(name, frequencies, thresholds))
            self._names.append(name)

        self._build_from_qualities(read_qualities)

    def rebuild(self, csv_file_path: str = ""):
        for name in self.names:
            del vars(self)[name]
        self._names = []
        self._is_built = False

        self.build(csv_file_path)

    def get_qualities_for(self, audio: np.ndarray) -> List[str]:
        assert self.is_built, "FBQualities must be built first"

        found_qualities = []
        spec = get_spectrogram(audio)

        for name in self.names:
            if eval(f"self.{name}.is_quality(audio, spec)"):
                found_qualities.append(name)
        return found_qualities
