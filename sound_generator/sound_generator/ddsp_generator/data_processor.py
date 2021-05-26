from typing import Dict
import numpy as np
from ddsp.core import midi_to_unit
from ddsp.training.preprocessing import F0LoudnessPreprocessor
from ..interfaces import DataProcessor


class DDSPDataProcessor(DataProcessor):
    """
    Prepares data to be fed into DDSP Generator
    """
    def process(self, inputs: Dict) -> Dict:
        f0_ld_inputs, f0_ld_outputs = inputs.get('f0_ld_inputs'), inputs.get('f0_ld_outputs')
        f0_scaled, ld_scaled = f0_ld_outputs
        target_pitch = inputs.get("target_pitch")

        if target_pitch is not None:
            # in this event, ignore the predicted f0_scaled
            f0_scaled = np.ones((1, 1000)) * midi_to_unit(target_pitch, midi_min=0, midi_max=127)
        f0_hz, loudness_db = F0LoudnessPreprocessor.invert_scaling(f0_scaled, ld_scaled)

        return {
            'z': f0_ld_inputs['latent_vector'],
            'loudness_db': loudness_db,
            'f0_hz': f0_hz
        }
