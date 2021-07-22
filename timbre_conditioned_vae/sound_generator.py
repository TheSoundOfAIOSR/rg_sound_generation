import os
import logging
import tensorflow as tf
import tsms
from typing import List, Dict
from tcvae import model, localconfig, predict
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class SoundGenerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SoundGenerator, cls).__new__(cls)
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
        self.encoder = None
        self.complete_model = None
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
        z = data.get("z") or np.random.randn(self.conf.latent_dim) * 5
        velocity = data.get("velocity") or 50
        pitch = data.get("pitch") or 60
        measures = data.get("measures") or np.random.randn(self.conf.num_measures)

        assert 25 <= velocity <= 127
        assert 40 <= pitch <= 88
        assert measures is not None

        z_out = np.expand_dims(z, axis=0)
        velocity_out = np.zeros((1, self.conf.num_velocities))
        velocity_out[:, int(velocity / 25) - 1] = 1.
        pitch -= self.conf.starting_midi_pitch
        pitch_out = np.zeros((1, self.conf.num_pitches))
        pitch_out[:, pitch] = 1.
        measures_out = np.expand_dims(measures, axis=0)

        assert z_out.shape == (1, self.conf.latent_dim)
        assert velocity_out.shape == (1, self.conf.num_velocities)
        assert pitch_out.shape == (1, self.conf.num_pitches)
        assert measures_out.shape == (1, self.conf.num_measures)

        return {
            "z_input": z_out,
            "velocity": velocity_out,
            "note_number": pitch_out,
            "measures": measures_out
        }

    def load_model(self, checkpoint_path: str = None) -> None:
        assert os.path.isfile(checkpoint_path), f"No checkpoint found at {checkpoint_path}"
        logger.info("Creating complete model from config")
        self.complete_model = model.get_model_from_config(self.conf)
        logger.info("Loading pretrained weights for complete model")
        self.complete_model.load_weights(checkpoint_path)
        logger.info("Complete model loaded")
        logger.info("Creating decoder")
        self.decoder = tf.keras.Model(
            self.complete_model.layers[-1].input, self.complete_model.layers[-1].output
        )
        self.decoder.trainable = False
        self.encoder = tf.keras.Model(
            self.complete_model.layers[1].input, self.complete_model.layers[1].output
        )
        logger.info("Decoder created")

    def get_prediction(self, data) -> Dict:
        processed_data = self.prepare_data(data)
        assert self.decoder is not None
        prediction = self.decoder.predict(processed_data)
        normalized_data_pred = self.conf.data_handler.output_transform(prediction, pred=True)
        note_number = np.argmax(processed_data["note_number"]) + self.conf.starting_midi_pitch
        mask = np.zeros((1, 1001, 110))
        f0 = tsms.core.midi_to_f0_estimate(note_number, 64, 64)
        harmonics = tsms.core.get_number_harmonics(f0, self.conf.sample_rate)
        harmonics = np.squeeze(harmonics.numpy())
        print(harmonics)
        mask[:, :, :harmonics + 1] = np.ones((1, 1001, harmonics + 1))
        h_freq_pred, h_mag_pred, h_phase_pred = self.conf.data_handler.denormalize(
            normalized_data_pred, mask, note_number)
        audio = tsms.core.harmonic_synthesis(
            h_freq_pred, h_mag_pred, h_phase_pred,
            self.conf.sample_rate, self.conf.frame_size
        )
        audio = np.squeeze(audio.numpy())
        return {
            "audio": audio.tolist(),
            "z": processed_data.get("z_input")[0].tolist(),
            "measures": processed_data.get("measures")[0].tolist()
        }
