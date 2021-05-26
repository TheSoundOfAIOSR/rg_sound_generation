from typing import Dict
import numpy as np
from ..interfaces import DataProcessor


class F0LoudnessDataProcessor(DataProcessor):
    def __init__(self, sequence_length: int=1000):
        self.sequence_length = sequence_length

    def _convert_to_sequence(self, inputs: np.ndarray) -> np.ndarray:
        dimension = inputs.shape[-1]
        outputs = np.zeros((self.sequence_length, dimension)) + np.squeeze(inputs)
        return np.expand_dims(outputs, axis=0).astype('float32')

    def process(self, inputs: Dict) -> Dict:
        z_inputs, z_outputs = inputs.get('z_inputs'), inputs.get('z_outputs')
        return {
            'pitch': self._convert_to_sequence(z_inputs.get('pitch')),
            'velocity': self._convert_to_sequence(z_inputs.get('velocity')),
            'instrument_source': self._convert_to_sequence(z_inputs.get('instrument_source')),
            'qualities': self._convert_to_sequence(z_inputs.get('qualities')),
            'latent_vector': z_outputs
        }
