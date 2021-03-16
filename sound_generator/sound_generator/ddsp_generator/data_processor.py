from ddsp.training.preprocessing import F0LoudnessPreprocessor


class DDSPDataProcessor:
    @staticmethod
    def process(inputs):
        f0_ld_inputs, f0_ld_outputs = inputs.get('f0_ld_inputs'), inputs.get('f0_ld_outputs')
        f0_scaled, ld_scaled = f0_ld_outputs
        f0_hz, loudness_db = F0LoudnessPreprocessor.invert_scaling(f0_scaled, ld_scaled)

        return {
            'z': f0_ld_inputs['latent_vector'],
            'loudness_db': loudness_db,
            'f0_hz': f0_hz
        }
