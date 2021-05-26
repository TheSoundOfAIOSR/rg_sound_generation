from typing import Dict
import tensorflow as tf
import ddsp.training
import gin
from loguru import logger
from ..interfaces import BaseModel


class DDSPGenerator(BaseModel):
    """
    DDSP Generator
    """
    def __init__(self, checkpoint_path: str, gin_config: str):
        self.model = None
        self.checkpoint_path = checkpoint_path
        self.gin_config = gin_config
        self._load_model()

    def _load_model(self):
        logger.info("Loading gin configuration")
        gin.parse_config_file(self.gin_config)
        self.model = ddsp.training.models.Autoencoder(encoder=None)
        logger.info("Loading model checkpoint")
        self.model.restore(self.checkpoint_path)

    def predict(self, inputs: Dict) -> tf.Tensor:
        logger.info("Fetching prediction")
        outputs = self.model(inputs, training=False)
        logger.info("Converting predicted outputs to audio")
        return self.model.get_audio_from_outputs(outputs)
