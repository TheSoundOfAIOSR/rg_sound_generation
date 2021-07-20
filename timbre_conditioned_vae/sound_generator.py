from tcvae import model, localconfig, predict
import numpy as np


class SoundGenerator:
    def __init__(self, config_path, data_handler_type):
        print("Initializing SoundGenerator")
        self.config_path = config_path
        self.conf = localconfig.LocalConfig(data_handler_type)
        self.conf.load_config_from_file(config_path)
        self.conf.batch_size = 1
        self.model = None
        assert data_handler_type == self.conf.data_handler_type, "Data handler type " \
                                                                 "does not match saved config"
        print("SoundGenerator initialized")

    def prepare_data(self, data):
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
        predict_single = data.get("predict_single") or True

    def load_model(self, checkpoint_path):
        print("Creating model from config")
        self.model = model.get_model_from_config(self.conf)
        print("Loading pretrained model weights")
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def get_prediction(self, data):
        self.prepare_data(data)
