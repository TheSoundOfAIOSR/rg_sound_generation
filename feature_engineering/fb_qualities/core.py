import os.path
from typing import List
import pandas as pd


class FBQuality:
    def __init__(self,
                 name: str,
                 frequencies: List[float]):
        assert len(frequencies) == 2, f"There must be exactly 2 values in " \
                               f"the list band, found {len(frequencies)}"
        assert frequencies[0] < frequencies[1], "First frequency in the band can not " \
                                  "be higher than second frequency"

        self._name = name
        self._frequencies = frequencies

    @property
    def name(self) -> str:
        return self._name

    @property
    def frequencies(self) -> List[float]:
        return self._frequencies


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

            read_qualities.append(FBQuality(name, frequencies))
            self._names.append(name)

        self._build_from_qualities(read_qualities)

    def rebuild(self, csv_file_path: str = ""):
        for name in self.names:
            del vars(self)[name]
        self._names = []
        self._is_built = False

        self.build(csv_file_path)

