import os
import logging
from typing import List, Dict
from tcvae import model, localconfig, predict
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class SoundGenerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SoundGenerator, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, config_path: str = None,
                 data_handler_type: str = "data_handler"):
        logger.info("Initializing SoundGenerator")
        if config_path is None:
            config_path = os.path.join(os.getcwd(), "checkpoints", "Default.json")
        self.config_path = config_path
        self.conf = localconfig.LocalConfig(data_handler_type)
        self.conf.load_config_from_file(config_path)
        self.conf.batch_size = 1
        self.decoder = None
        assert data_handler_type == self.conf.data_handler_type, "Data handler type " \
                                                                 "does not match saved config"
        logger.info("SoundGenerator initialized")

    def prepare_data(self, data: Dict) -> Dict:
        """
        z: List of conf.latent_dim values
        velocity: int between [25, 127]
        pitch: int between [40, 88]
        heuristics: List of conf.num_heuristics values
        predict_single: bool, True if returning 1 value for the given pitch
            False if returning entire pitch range from 40 to 88
        """
        z = data.get("z") or np.random.randn(self.conf.latent_dim)
        velocity = data.get("velocity") or 50
        pitch = data.get("pitch") or 60
        heuristics = data.get("heuristics")

        assert 25 <= velocity <= 127
        assert 40 <= pitch <= 88
        assert heuristics is not None

        z_out = np.expand_dims(z, axis=0)
        velocity_out = np.zeros((1, self.conf.num_velocities))
        velocity_out[:, int(velocity / 25) - 1] = 1.
        pitch -= self.conf.starting_midi_pitch
        pitch_out = np.zeros((1, self.conf.num_pitches))
        pitch_out[:, pitch] = 1.
        heuristics_out = np.expand_dims(heuristics, axis=0)

        assert z_out.shape == (1, self.conf.latent_dim)
        assert velocity_out.shape == (1, self.conf.num_velocities)
        assert pitch_out.shape == (1, self.conf.num_pitches)
        assert heuristics_out.shape == (1, self.conf.num_measures)

        return {
            "z_input": z_out,
            "velocity": velocity_out,
            "note_number": pitch_out,
            "measures": heuristics_out
        }

    def load_model(self, checkpoint_path: str = None) -> None:
        assert os.path.isfile(checkpoint_path), f"No checkpoint found at {checkpoint_path}"
        logger.info("Creating complete model from config")
        complete_model = model.get_model_from_config(self.conf)
        logger.info("Loading pretrained weights for complete model")
        complete_model.load_weights(checkpoint_path)
        logger.info("Complete model loaded")
        logger.info("Creating decoder")

    def get_prediction(self, data) -> (List[float], bool):
        self.prepare_data(data)
        return [], True
