import os
import tensorflow as tf
import tsms
from typing import Dict
from tcvae import model, localconfig
from tcvae.compute_measures import heuristic_names
import numpy as np


class SoundGenerator:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SoundGenerator, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path: str = None):
        assert os.path.isfile(config_path)
        self.config_path = config_path
        self.conf = localconfig.LocalConfig()
        self.conf.load_config_from_file(config_path)
        self.model = None
        # Prediction specific config
        self.conf.batch_size = 1
        self.measure_to_index = dict((n, i) for i, n in enumerate(heuristic_names))
        self.index_to_measure = dict((v, k) for k, v in self.measure_to_index.items())

    def load_model(self, checkpoint_path: str = None) -> None:
        assert os.path.isfile(checkpoint_path), f"No checkpoint at {checkpoint_path}"
        self.model = model.MtVae(self.conf)
        # ToDo - Add input shapes
        self.model.build([])
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def _get_mask(self, note_number):
        f0 = tsms.core.midi_to_f0_estimate(note_number, self.conf.frame_size,
                                           self.conf.frame_size)
        harmonics = tsms.core.get_number_harmonics(f0, self.conf.sample_rate)
        harmonics = np.squeeze(harmonics)
        mask = np.zeros((1, self.conf.harmonic_frame_steps, 110))
        mask[:, :, :harmonics] = np.ones((1, self.conf.harmonic_frame_steps, harmonics))
        return mask

    def _prepare_params(self, params: Dict) -> Dict:
        output = {}

        note_number = params.get("note_number") or 60
        velocity = params.get("velocity") or 50
        measures = params.get("measures") or {}

        assert 40 <= note_number <= 88
        note_number -= self.conf.starting_midi_pitch
        updated_note = np.zeros((1, self.conf.num_pitches))
        updated_note[:, note_number] = 1.
        output["note_number"] = updated_note

        assert 25 <= velocity <= 127
        velocity = int(velocity / 25 - 1)
        updated_vel = np.zeros((1, self.conf.num_velocities))
        updated_vel[:, velocity] = 1.
        output["velocity"] = updated_vel

        updated_measures = [0.2] * self.conf.num_measures

        for m, val in measures.items():
            assert m in heuristic_names
            updated_measures[self.measure_to_index[m]] = val

        output["measures"] = np.expand_dims(updated_measures, axis=0)

        if "z" in params:
            updated_z = params.get("z")
            for val in updated_z:
                assert 0 <= val <= 1
            updated_z = np.expand_dims(updated_z, axis=0)
            assert updated_z.shape == (1, 16)
            output["z"] = updated_z
        else:
            print("Updating z from random values")
            output["z"] = np.random.rand(1, 16)
        return output

    def _get_prediction(self, params: Dict, prediction: tf.Tensor) -> np.ndarray:
        params = params.copy()

        note_number = np.argmax(params["note_number"], axis=-1) + self.conf.starting_midi_pitch
        transform = self.conf.data_handler.output_transform(prediction, pred=True)
        mask = self._get_mask(note_number)

        h_freq, h_mag, h_phase = self.conf.data_handler.denormalize(
            transform, mask, note_number)
        audio = tsms.core.harmonic_synthesis(
            h_freq, h_mag, h_phase, self.conf.sample_rate, self.conf.frame_size)
        return np.squeeze(audio.numpy())

    def get_prediction(self, params: Dict) -> np.ndarray:
        params = params.copy()

        params = self._prepare_params(params)
        prediction = self.model.decoder.predict(params)
        audio_pred = self._get_prediction(params, prediction=prediction)

        return audio_pred
