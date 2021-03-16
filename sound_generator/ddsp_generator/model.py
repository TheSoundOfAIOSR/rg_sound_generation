import ddsp.training
import gin


class DDSPGenerator:
    def __init__(self, checkpoint_path: str, gin_config: str):
        self.model = None
        self.checkpoint_path = checkpoint_path
        self.gin_config = gin_config
        self._load_model()

    def _load_model(self):
        gin.parse_config_file(self.gin_config)
        self.model = ddsp.training.models.Autoencoder(encoder=None)
        self.model.restore(self.checkpoint_path)

    def predict(self, inputs):
        outputs = self.model(inputs, training=False)
        return self.model.get_audio_from_outputs(outputs)
