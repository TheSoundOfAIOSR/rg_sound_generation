from typing import List
from .config import Config


fb_qualities = {
    "weak":       {"range": [20, 80],      "amp_class": 0}, # a bit irrelevant to guitar samples in normal range
    "thin":       {"range": [20, 400],     "amp_class": 0,   "r_thres_high": 0.08  },
    "hollow":     {"range": [140, 1500],   "amp_class": 0,   "r_thres_high": 0.39  },
    "distant":    {"range": [1500, 7500],  "amp_class": 0}, # too similar to dark
    "dark":       {"range": [2000, 20000], "amp_class": 0,   "r_thres_high": 0.20 },
    "dull":       {"range": [7500, 20000], "amp_class": 0}, # irrelevant this can't be represented by 16k sr
    "bottom":     {"range": [20, 100],     "amp_class": 1}, # a bit irrelevant to guitar samples in normal range
    "warm":       {"range": [90, 450],     "amp_class": 1,   "r_thres_low": 0.31},
    "punch":      {"range": [90, 200],     "amp_class": 1}, # too small a range to figure from current spec
    "clear":      {"range": [90, 300],     "amp_class": 1}, # too small a range to figure from current spec
    "full":       {"range": [200, 1000],   "amp_class": 1,   "r_thres_low": 0.60},
    "edge":       {"range": [1000, 4000],  "amp_class": 1},
    "bright":     {"range": [1000, 20000], "amp_class": 1,   "r_thres_low": 0.47}, # also "presence"
    "air":        {"range": [8000, 20000], "amp_class": 1}, # irrelevant this can't be represented by 16k sr
    "rumble":     {"range": [20, 50],      "amp_class": 2}, # a bit irrelevant to guitar samples in normal range
    "boomy":      {"range": [40, 120],     "amp_class": 2}, # a bit irrelevant to guitar samples in normal range
    "muddy":      {"range": [90, 450],     "amp_class": 2,   "r_thres_low": 0.46 }, # a bit hard to say
    "boxy":       {"range": [300, 600],    "amp_class": 2,   "r_thres_low": 0.34 },
    "honky":      {"range": [500, 1200],   "amp_class": 2,   "r_thres_low": 0.43 }, # also called "nasaly"
    "harsh":      {"range": [1100, 8000],  "amp_class": 2,   "r_thres_low": 0.65 },
    "tinny":      {"range": [1000, 1800],  "amp_class": 2,   "r_thres_low": 0.24 },
    "sibilance":  {"range": [3500, 10000], "amp_class": 2,   "r_thres_low": 0.21 },
    "brittle":    {"range": [8000, 20000], "amp_class": 2}, # irrelevant this can't be represented by 16k sr
    "piercing":   {"range": [8000, 15000], "amp_class": 2}, # irrelevant this can't be represented by 16k sr
}


def hz_to_band(frequencies: List) -> List:
    factor = Config.sample_rate / Config.fft_len
    max_ = int(0.5 * Config.sample_rate / factor) + 1
    f1, f2 = frequencies
    f1, f2 = int(f1 / factor), min(max_, int(f2 / factor))
    return [f1, f2]
