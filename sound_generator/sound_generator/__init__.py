"""
Sound Generator
"""
import os
from typing import Dict
import numpy as np
from loguru import logger
from .generator import SoundGenerator
from .validation import validate_input


EMPTY_RESULT = np.zeros((1, ))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_sound_generator = SoundGenerator()
_sound_generator.load_models()


def load_models() -> bool:
    global _sound_generator
    if not _sound_generator.models_loaded:
        _sound_generator.load_models()
    return _sound_generator.models_loaded


def get_prediction(inputs: Dict) -> (np.ndarray, bool):
    """
    Generates and returns audio as a numpy array
    :param inputs: Dict
        latent_sample: must be a list of 16 floating point values between -7 and +7
        velocity: can be one of [25, 50, 75, 100, 127]
        pitch: must be between 9 and 120
        source: can be one of ["acoustic", "electronic", "synthetic"]
        qualities: can be a have a number of qualities from
            ["bright", "dark", "distortion", "fast_decay", "long_release",
            "multiphonic", "nonlinear_env", "percussive", "reverb", "tempo_sync"]
    :return: np.ndarray, bool
    """
    global _sound_generator

    if not validate_input(inputs):
        logger.error("Could not validate inputs")
        return EMPTY_RESULT, False

    if not load_models():
        logger.error("Could not load models")
        return EMPTY_RESULT, False

    return _sound_generator.predict(inputs)
