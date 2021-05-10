"""
        latent_sample: must be a list of 16 floating point values between -7 and +7
        velocity: can be one of [25, 50, 75, 100, 127]
        pitch: must be between 9 and 120
        source: can be one of ["acoustic", "electronic", "synthetic"]
        qualities: can be a have a number of qualities from
            ["bright", "dark", "distortion", "fast_decay", "long_release",
            "multiphonic", "nonlinear_env", "percussive", "reverb", "tempo_sync"]
"""
from typing import Dict, List
from loguru import logger


def validate_input(inputs: Dict) -> bool:
    expected_keys = [
        "latent_sample",
        "velocity",
        "pitch",
        "source",
        "qualities"
    ]
    expected_qualities = [
        "bright", "dark", "distortion",
        "fast_decay", "long_release",
        "multiphonic", "nonlinear_env",
        "percussive", "reverb", "tempo_sync"
    ]
    expected_sources = ["acoustic", "electronic", "synthetic"]
    expected_velocities = [25, 50, 75, 100, 127]

    try:
        for key in inputs.keys():
            assert key in expected_keys, f"Unexpected key = '{key}' in inputs"

        latent_sample = inputs.get("latent_sample")
        if latent_sample is not None:
            assert isinstance(latent_sample, list), f"latent_sample must be a list, not {type(latent_sample)}"
            for val in latent_sample:
                assert type(val) == int or type(val) == float, "latent_sample values " \
                                                               "must be either float or int"
                assert val < -7. or val > 7., "latent_sample values must be between -7 and +7"

        velocity = inputs.get("velocity")
        if velocity is not None:
            assert type(velocity) == int, "velocity must be of type int"
            assert velocity in expected_velocities, f"velocity must be one of " \
                                                    f"following values {expected_velocities}"

        pitch = inputs.get("pitch")
        if pitch is not None:
            assert type(pitch) == int, "pitch must be of type int"
            assert 9 <= pitch <= 120, "pitch value can only be between 9 and 120"

        source = inputs.get("source")
        if source is not None:
            assert type(source) == str, "source must be of type str"
            assert source in expected_sources, f"source must be one of {expected_sources}"

        qualities = inputs.get("qualities")
        if qualities is not None:
            assert isinstance(qualities, list), f"qualities must be a list, not {type(qualities)}"
            for q in qualities:
                assert q in expected_qualities, f"qualities must be one of {expected_qualities}"

        return True
    except Exception as e:
        logger.info("Validation failed")
        logger.error(e)
    return False
