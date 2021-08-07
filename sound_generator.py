import os
import tensorflow as tf
from tensorflow import TensorShape
import tsms
import numpy as np
from urllib.request import urlretrieve
from loguru import logger
from pprint import pprint
import warnings
from typing import Dict, Any
from tcae import model, localconfig
from tcae.dataset import heuristic_names


warnings.simplefilter("ignore")


# Todo: Sound Generation prediction / output transform needs to be fixed


def get_zero_batch(conf: localconfig.LocalConfig):
    mask = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_num_harmonics])
    note_number = TensorShape([conf.batch_size, conf.num_pitches])
    velocity = TensorShape([conf.batch_size, conf.num_velocities])
    measures = TensorShape([conf.batch_size, conf.num_measures])
    f0_shifts = TensorShape([conf.batch_size, conf.harmonic_frame_steps, 1])
    mag_env = TensorShape([conf.batch_size, conf.harmonic_frame_steps, 1])
    h_freq_shifts = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_num_harmonics])
    h_mag_dist = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_num_harmonics])
    h_phase_diff = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_num_harmonics])

    _shapes = {}

    _shapes.update({
        "mask": tf.zeros(mask),
        "f0_shifts": tf.zeros(f0_shifts),
        "mag_env": tf.zeros(mag_env),
        "h_freq_shifts": tf.zeros(h_freq_shifts),
        "h_mag_dist": tf.zeros(h_mag_dist),
        "h_phase_diff": tf.zeros(h_phase_diff)
    })

    if conf.use_note_number:
        _shapes.update({"note_number": tf.zeros(note_number)})
    if conf.use_velocity:
        _shapes.update({"velocity": tf.zeros(velocity)})
    if conf.use_heuristics:
        _shapes.update({"measures": tf.zeros(measures)})
    return _shapes


class SoundGenerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SoundGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = None,
                 checkpoint_path: str = None,
                 auto_download: bool = True):

        tf.keras.backend.clear_session()

        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._conf = localconfig.LocalConfig()
        self._model = None

        # Use defaults
        if self._config_path is None:
            default_config_path = os.path.join(os.getcwd(), "deployed", "conf.json")
            if os.path.isfile(default_config_path):
                logger.info("Using default config")
                self._config_path = default_config_path

        if self._checkpoint_path is None:
            default_checkpoint_path = os.path.join(os.getcwd(), "deployed", "model.h5")
            if not os.path.isfile(default_checkpoint_path) and auto_download:
                logger.info("Downloading default model checkpoint")
                _ = urlretrieve("https://osr-tsoai.s3.amazonaws.com/mt_5/model.h5", "deployed/model.h5")
            if os.path.isfile(default_checkpoint_path):
                logger.info("Using default model checkpoint")
                self._checkpoint_path = default_checkpoint_path

        if self._config_path is not None:
            self.load_config()

        if self._checkpoint_path is not None:
            self.load_model()

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @checkpoint_path.setter
    def checkpoint_path(self, new_path: str):
        self._checkpoint_path = new_path

    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, new_path: str):
        self._config_path = new_path

    @property
    def conf(self):
        return self._conf

    @property
    def model(self):
        return self._model

    def load_config(self) -> None:
        assert os.path.isfile(self._config_path), f"No config at {self._config_path}"
        self._conf.load_config_from_file(self._config_path)
        # Prediction specific config
        self._conf.batch_size = 1
        self._conf.print_model_summary = False
        logger.info("Config loaded")

    def load_model(self) -> None:
        assert os.path.isfile(self._checkpoint_path), f"No checkpoint at {self._checkpoint_path}"
        self._model = model.TCAEModel(self._conf)
        _ = self._model(get_zero_batch(self._conf))
        self._model.load_weights(self._checkpoint_path)
        logger.info("Model loaded")

    def _get_mask(self, note_number: int) -> np.ndarray:
        f0 = tsms.core.midi_to_f0_estimate(
            note_number, self._conf.frame_size,
            self._conf.frame_size
        )
        harmonics = tsms.core.get_number_harmonics(f0, self._conf.sample_rate)
        harmonics = np.squeeze(harmonics)
        mask = np.zeros((1, self._conf.harmonic_frame_steps, self._conf.max_num_harmonics))
        mask[:, :, :harmonics] = np.ones((1, self._conf.harmonic_frame_steps, harmonics))
        return mask

    def _prepare_note_number(self, note_number) -> np.ndarray:
        index = note_number - self._conf.starting_midi_pitch
        encoded = np.zeros((self._conf.num_pitches, ))
        encoded[index] = 1.
        return encoded

    def _prepare_velocity(self, velocity) -> np.ndarray:
        index = velocity // 25 - 1
        encoded = np.zeros((self._conf.num_velocities, ))
        encoded[index] = 1.
        return encoded

    def _prepare_inputs(self, data: Dict) -> Dict:
        logger.info("Preparing inputs")

        output_note_number = data.get("pitch") or 60
        input_note_number = data.get("input_pitch") or 60
        velocity = data.get("velocity") or 75
        latent_sample = data.get("latent_sample") or np.random.rand(self._conf.latent_dim)
        heuristic_measures = data.get("heuristic_measures") or np.random.rand(self._conf.num_measures)
        _ = data.get("qualities") or []

        assert 40 <= input_note_number <= 88, "Conditioning note number must be between" \
                                              " 40 and 88"
        assert 25 <= velocity <= 127, "Velocity must be between 25 and 127"
        assert np.shape(latent_sample) == (self._conf.latent_dim, )
        assert np.shape(heuristic_measures) == (self._conf.num_measures, )

        # Heuristic measures will be updated according to qualities present

        mask = self._get_mask(output_note_number)

        decoder_inputs = {
            "mask": mask,
            "note_number": np.expand_dims(self._prepare_note_number(input_note_number), axis=0),
            "velocity": np.expand_dims(self._prepare_velocity(velocity), axis=0),
            "z": np.expand_dims(latent_sample, axis=0),
            "measures": np.expand_dims(np.array(heuristic_measures), axis=0)
        }
        return decoder_inputs

    def info(self) -> None:
        print("=" * 40)
        print("Expected input dictionary:")
        print("=" * 40)
        pprint({
            "input_pitch": 40,
            "pitch": 40,
            "velocity": 100,
            "heuristic_measures": np.random.rand(self._conf.num_measures).tolist(),
            "latent_sample": np.random.rand(self._conf.latent_dim).tolist(),
            "qualities": []
        })
        print("=" * 40)
        print("input_pitch: Note number to use in decoder input")
        print("pitch: Note number to use in audio synthesis")
        print("velocity: Velocity of the note between 25 and 127")
        print("heuristic_measures: List of values for following measures used in decoder in the sequence shown:")
        pprint(heuristic_names)
        print("latent_sample: Values for z input to decoder")
        print("qualities: List of words detected from user speech")
        print("=" * 40)

    def get_prediction(self, data: Dict) -> Any:
        try:
            output_note_number = data.get("pitch") or 60
            decoder_inputs = self._prepare_inputs(data)
            logger.info("Getting prediction")
            prediction = self._model.decoder(decoder_inputs)
            logger.info("Transforming prediction")
            transformed = self._conf.data_handler.output_transform(prediction)
            logger.info("De-normalizing prediction")
            freq, mag, phase = self._conf.data_handler.denormalize(
                transformed, decoder_inputs["mask"],
                output_note_number
            )
            logger.info("Synthesising audio")
            audio = tsms.core.harmonic_synthesis(
                freq, mag, phase,
                self._conf.sample_rate,  self._conf.frame_size
            )
            return True, np.squeeze(audio).tolist()
        except Exception as e:
            logger.error(e)
        return False, []
