import numpy as np


class ZDataProcessor:
    def __init__(self):
        self.source_to_index = {
            'acoustic': 0,
            'electronic': 1,
            'synthetic': 2
        }
        self.quality_to_index = {
            'bright': 0,
            'dark': 1,
            'distortion': 2,
            'fast_decay': 3,
            'long_release': 4,
            'multiphonic': 5,
            'nonlinear_env': 6,
            'percussive': 7,
            'reverb': 8,
            'tempo_sync': 9
        }

    def process(self, inputs):
        velocity = inputs.get('velocity') or 75
        pitch = inputs.get('velocity') or 60
        source = inputs.get('source') or 'acoustic'
        qualities = inputs.get('qualities') or []
        latent_sample = inputs.get('latent_sample') or [0.] * 16

        velocity = np.expand_dims([velocity / 127.], axis=0).astype('float32')
        pitch = np.expand_dims([pitch / 127.], axis=0).astype('float32')
        source = np.expand_dims([self.source_to_index[source] / 2.], axis=0).astype('float32')
        latent_sample = np.expand_dims(latent_sample, axis=0).astype('float32')

        qualities_one_hot = np.zeros((1, 10))

        for _, q in enumerate(qualities):
            qualities_one_hot[0, self.quality_to_index[q]] = 1.

        return {
            'velocity': velocity,
            'pitch': pitch,
            'instrument_source': source,
            'qualities': qualities_one_hot.astype('float32'),
            'z_input': latent_sample
        }
