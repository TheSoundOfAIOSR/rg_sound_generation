from typing import List, Dict
import numpy as np
from .spectrogram import get_audio, get_spectrogram
from .core import fb_qualities, hz_to_band


def find_fb_qualities(file_path: str,
                      raw_values: bool = False,
                      return_all: bool = False) -> (List, Dict):
    """
    Find freqyency based qualities for a given audio sample
    Args:
        file_path: string file path to the audio sample
        raw_values: if the raw values calculated based on frequency based
            qualities' rules is to be returned
        return_all: if all the raw values are to be returned as opposed
            to just the ones that are found in the sample

    Returns:
        (List, Dict) - a list of found qualities, and a dictionary of raw
        values if the raw_values argument is set to True
    """
    audio = get_audio(file_path)
    spec = get_spectrogram(audio)

    found = []
    raw = {}

    for q in fb_qualities.keys():
        r = fb_qualities[q]["range"]
        slice_range = hz_to_band(r)
        sliced_spec = spec[slice_range[0]: slice_range[1], :]
        ratio = np.sum(sliced_spec) / np.sum(spec)

        high_thres = fb_qualities[q].get("r_thres_high") or 1.0
        low_thres = fb_qualities[q].get("r_thres_low") or 0.

        if high_thres == 1 and low_thres == 0:
            continue

        if return_all:
            raw[q] = {
                "ratio": ratio,
                "r_thres_high": high_thres,
                "r_thres_low": low_thres
            }

        if low_thres < ratio < high_thres:
            found.append(q)
            if raw_values and q not in raw:
                raw[q] = {
                    "ratio": ratio,
                    "r_thres_high": high_thres,
                    "r_thres_low": low_thres
                }

    return append_keys_with_fb(found, raw)


def append_keys_with_fb(found: List, raw: Dict) -> (List, Dict):
    updated_found = [f"fb_{f}" for f in found]
    updated_raw = dict((f"fb_{key}", value) for key, value in raw.items())
    return updated_found, updated_raw
