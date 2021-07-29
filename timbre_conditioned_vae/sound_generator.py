import os
import tsms
import logging
import warnings
from typing import Dict, Any
from tcvae import model, localconfig, train
from tcvae.compute_measures import heuristic_names
import numpy as np


warnings.simplefilter("ignore")

logger = logging.getLogger("sound_gen_logger")
logger.setLevel(logging.INFO)


class HeuristicMeasures:
    def __init__(self):
        self.names = heuristic_names

        for name in heuristic_names:
            vars(self)[name] = 0.

    def __call__(self, **kwargs):
        outputs = []

        for k, v in kwargs.items():
            if k in vars(self):
                vars(self)[k] = v

        for name in heuristic_names:
            outputs.append(vars(self)[name])
        return outputs


class SoundGenerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SoundGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = None,
                 checkpoint_path: str = None):
        self._config_path = config_path
        self._checkpoint_path = checkpoint_path
        self._conf = localconfig.LocalConfig()
        self._model = None
        self._measure_to_index = dict((n, i) for i, n in enumerate(heuristic_names))
        self._index_to_measure = dict((v, k) for k, v in self._measure_to_index.items())

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

    def load_config(self) -> None:
        assert os.path.isfile(self._config_path), f"No config at {self._config_path}"
        self._conf.load_config_from_file(self._config_path)
        # Prediction specific config
        self._conf.batch_size = 1
        logger.info("Config loaded")

    def load_model(self) -> None:
        assert os.path.isfile(self._checkpoint_path), f"No checkpoint at {self._checkpoint_path}"
        self._model = model.MtVae(self._conf)
        # For some reason model.build doesn't work
        _ = self._model(train.get_zero_batch(self._conf))
        self._model.load_weights(self._checkpoint_path)
        logger.info("Model loaded")

    def _get_mask(self, note_number: int) -> np.ndarray:
        f0 = tsms.core.midi_to_f0_estimate(
            note_number, self._conf.frame_size,
            self._conf.frame_size
        )
        harmonics = tsms.core.get_number_harmonics(f0, self._conf.sample_rate)
        harmonics = np.squeeze(harmonics)
        mask = np.zeros((1, self._conf.harmonic_frame_steps, 110))
        mask[:, :, :harmonics] = np.ones((1, self._conf.harmonic_frame_steps, harmonics))
        return mask

    def _prepare_note_number(self, note_number):
        index = note_number - self._conf.starting_midi_pitch
        encoded = np.zeros((self._conf.num_pitches, ))
        encoded[index] = 1.
        return encoded

    def _prepare_velocity(self, velocity):
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

        assert 40 <= input_note_number <= 88, "Conditioning note number must be between" \
                                              " 40 and 88"
        assert 25 <= velocity <= 127, "Velocity must be between 25 and 127"
        assert np.shape(latent_sample) == (self._conf.latent_dim, )
        assert np.shape(heuristic_measures) == (self._conf.num_measures, )

        mask = self._get_mask(output_note_number)

        decoder_inputs = {
            "mask": mask,
            "note_number": np.expand_dims(self._prepare_note_number(input_note_number), axis=0),
            "velocity": np.expand_dims(self._prepare_velocity(velocity), axis=0),
            "z": np.expand_dims(latent_sample, axis=0),
            "measures": np.expand_dims(np.array(heuristic_measures), axis=0)
        }
        return decoder_inputs

    def get_prediction(self, data: Dict) -> Any:
        try:
            output_note_number = data.get("pitch") or 60
            decoder_inputs = self._prepare_inputs(data)
            logger.info("Getting prediction")
            prediction = self._model.decoder(decoder_inputs)
            logger.info("Transforming prediction")
            transformed = self._conf.data_handler.prediction_transform(prediction)
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
