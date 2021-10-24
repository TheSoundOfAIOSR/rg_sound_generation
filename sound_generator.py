import os
import tensorflow as tf
import tsms
import json
import numpy as np
import random
import zipfile
import pickle
import gdown
import warnings
from loguru import logger
from typing import Dict, Any, List
from tensorflow import TensorShape
from tcae import model, train, localconfig


tf.get_logger().setLevel("ERROR")
warnings.simplefilter("ignore")
deployed_dir = os.path.join(os.getcwd(), "deployed")

if not os.path.isdir(deployed_dir):
    logger.info("Deployed dir not found, creating")
    os.mkdir(deployed_dir)


def get_zero_batch(conf: localconfig.LocalConfig):
    mask = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_num_harmonics])
    note_number = TensorShape([conf.batch_size, 1])
    velocity = TensorShape([conf.batch_size, 1])
    measures = TensorShape([conf.batch_size, conf.num_measures])
    f0_shifts = TensorShape([conf.batch_size, conf.harmonic_frame_steps, 1])
    mag_env = TensorShape([conf.batch_size, conf.harmonic_frame_steps, 1])
    h_freq_shifts = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_num_harmonics])
    h_mag_dist = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_num_harmonics])
    h_freq_correction = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_num_harmonics])

    _shapes = {}

    _shapes.update({
        "mask": tf.zeros(mask),
        "f0_shifts": tf.zeros(f0_shifts),
        "mag_env": tf.zeros(mag_env),
        "h_freq_shifts": tf.zeros(h_freq_shifts),
        "h_mag_dist": tf.zeros(h_mag_dist),
        "h_freq_correction": tf.zeros(h_freq_correction),
        "instrument_id": tf.zeros([conf.batch_size, conf.num_instruments]),
        "name": tf.convert_to_tensor([b"a"] * conf.batch_size, dtype=tf.string)
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
        self._decoder_inputs = None
        self._mapping_data = None
        self._instrument_id_options = (
            0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 15, 16, 18, 19, 20, 21, 22,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73
        )
        self._taxonomy = (
            "bright", "dark", "smooth", "rough", "pure", "noisy",
            "clear", "muddy", "warm", "metallic", "full", "hollow",
            "thick", "thin", "rich", "sparse", "soft", "hard"
        )
        self._tax_to_index = dict((k, i) for i, k in enumerate(self._taxonomy))
        self._index_to_tax = dict((v, k) for k, v in self._tax_to_index.items())

        if self._checkpoint_path is None:
            default_checkpoint_path = os.path.join(deployed_dir, "new_data_132_0.00961.ckpt")
            default_checkpoint_file_path = f"{default_checkpoint_path}.index"
            if not os.path.isfile(default_checkpoint_file_path) and auto_download:
                download_path = os.path.join(deployed_dir, "model.zip")

                if not os.path.isfile(download_path):
                    logger.info("Downloading default model checkpoint")
                    download_url = "https://drive.google.com/uc?id=1Oe_GHDa6efwXuJUl5ZFP6tLZRV9q6Gea"
                    gdown.download(download_url, download_path, quiet=False)
                    logger.info("Model checkpoint downloaded")

                logger.info("Extracting archive")

                with zipfile.ZipFile(download_path) as zf:
                    zf.extractall(deployed_dir)

            if os.path.isfile(default_checkpoint_file_path):
                logger.info("Using default model checkpoint")
                self._checkpoint_path = default_checkpoint_path

        # Use defaults
        if self._config_path is None:
            default_config_path = os.path.join(deployed_dir, "conf.txt")
            if os.path.isfile(default_config_path):
                logger.info("Using default config")
                self._config_path = default_config_path

        # Config should be loaded before the model
        if self._config_path is not None:
            self.load_config()
        else:
            logger.warning("No config path set yet. Make sure this is not a mistake")

        if self._checkpoint_path is not None:
            self.load_model()
        else:
            logger.warning("No checkpoint path set yet. Make sure this is not a mistake")

        if self._mapping_data is None:
            logger.info("Loading qualities mapping data")
            with open("assets/mapping_data.json", "r") as f:
                self._mapping_data = json.load(f)

        self._load_decoder_values()

    @property
    def decoder_inputs(self):
        return self._decoder_inputs

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
    def instrument_id_options(self):
        return self._instrument_id_options

    @property
    def conf(self):
        return self._conf

    @property
    def model(self):
        return self._model

    @property
    def measures_names(self):
        return self._conf.data_handler.measure_names

    def load_config(self) -> None:
        assert os.path.isfile(self._config_path), f"No config at {self._config_path}"
        self._conf.load_config_from_file(self._config_path)
        # Prediction specific config
        self._conf.batch_size = 1
        self._conf.print_model_summary = False
        logger.info("Config loaded")

    def load_model(self) -> None:
        # assert os.path.isfile(self._checkpoint_path), f"No checkpoint at {self._checkpoint_path}"
        model_wrapper = train.ModelWrapper(model.TCAEModel(self._conf), self._conf.data_handler.loss)
        _ = model_wrapper(get_zero_batch(self._conf))
        model_wrapper.load_weights(self._checkpoint_path)
        self._model = model_wrapper.model
        self._model.trainable = False
        logger.info("Model loaded")

    def _qualities_embedding(self, qualities: List[str]) -> List[int]:
        all_words, emb = [], []

        for word in qualities:
            all_words += str(word).lower().replace(" ", "").split(",")

        for word in all_words:
            if word == "":
                continue
            assert word in self._tax_to_index, f"{word} not found in taxonomy"
            emb.append(self._tax_to_index[word])

        return list(sorted(emb))

    def _find_iids(self, qualities: List[str]) -> List[int]:
        emb = self._qualities_embedding(qualities)
        matches = dict((k, 0) for k in self._mapping_data.keys())

        for key, value in self._mapping_data.items():
            for index in value:
                if index in emb:
                    matches[key] += 1

        count_to_ids = {}

        for key, count in matches.items():
            if count not in count_to_ids:
                count_to_ids[count] = [int(key)]
            else:
                count_to_ids[count].append(int(key))

        max_count = max(count_to_ids.keys())
        matching_ids = count_to_ids[max_count]

        if len(matching_ids) == 0:
            return list(self._instrument_id_options)
        return matching_ids

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

    def get_measures_mean(self, input_pitch: int, velocity: int) -> (Dict, int, int):
        note_index = input_pitch - self.conf.starting_midi_pitch
        velocity_index = velocity // 25 - 1
        measures_mean = self._conf.data_handler.get_measures_mean(
            note_index, velocity_index
        )
        return measures_mean, note_index, velocity_index

    def get_decoder_index(self, note_index: int, velocity_index: int, instrument_index: int) -> int:
        c0 = self._conf.num_pitches
        c1 = c0 * self._conf.num_velocities
        index = note_index + c0 * velocity_index + c1 * instrument_index
        return index

    @staticmethod
    def measure_transform(measure_value: float, measure_mean: float) -> float:
        measure_value = 2.0 * measure_value - 1.0

        if measure_value >= 0.0:
            measure_value = measure_mean + measure_value * (1.0 - measure_mean)
        else:
            measure_value = (1.0 + measure_value) * measure_mean
        return measure_value

    @staticmethod
    def inverse_measure_transform(measure_value: float, measure_mean: float) -> float:
        if measure_value >= measure_mean:
            measure_value = (measure_value - measure_mean) / (1.0 - measure_mean)
        else:
            measure_value = (measure_value - measure_mean) / measure_mean

        measure_value = (measure_value + 1.0) / 2.0
        return measure_value

    def _load_decoder_values(self):
        assert os.path.isfile("assets/decoder_inputs.pickle")

        logger.info("Loading decoded inputs")

        with open("assets/decoder_inputs.pickle", "rb") as f:
            decoder_inputs = pickle.load(f)
        self._decoder_inputs = decoder_inputs

    def _prepare_instrument_id(self, instrument_id: int) -> np.ndarray:
        encoded = np.array([instrument_id], dtype=np.float32)
        # encoded[instrument_id] = 1.
        return encoded * self.conf.num_instruments

    def _prepare_note_number(self, note_number) -> np.ndarray:
        index = note_number - self._conf.starting_midi_pitch
        encoded = float(index) / self._conf.num_pitches
        # encoded = np.zeros((self._conf.num_pitches, ))
        # encoded[index] = 1.
        return encoded

    def _prepare_velocity(self, velocity) -> np.ndarray:
        index = velocity // 25 - 1
        encoded = float(index) / self._conf.num_velocities
        # encoded = np.zeros((self._conf.num_velocities, ))
        # encoded[index] = 1.
        return encoded

    def load_preset_fn(self, measures_mean: Dict, note_index: int,
                       velocity_index: int, instrument_id: int) -> Any:
        decoder_index = self.get_decoder_index(note_index, velocity_index, instrument_id)
        decoder_value = self.decoder_inputs[decoder_index]
        if decoder_value["z"] is not None and decoder_value["measures"] is not None:
            logger.info("Updating latent sample")
            latent_sample = decoder_value["z"].numpy()[0]
            logger.info("Updating measures")
            heuristic_measures = [
                self.inverse_measure_transform(v, measures_mean[k])
                for k, v in zip(self._conf.data_handler.measure_names, decoder_value["measures"].numpy()[0])
            ]
            return True, (latent_sample, heuristic_measures)
        return False, (None, None)

    def _prepare_inputs(self, data: Dict) -> Dict:
        logger.info("Preparing inputs")

        load_preset = data.get("load_preset") or False
        instrument_id = data.get("instrument_id") or 0
        output_note_number = data.get("pitch") or 60
        input_note_number = data.get("input_pitch") or output_note_number
        velocity = data.get("velocity") or 75
        latent_sample = data.get("latent_sample") or np.random.rand(self._conf.latent_dim)
        heuristic_measures = data.get("heuristic_measures") or np.random.rand(self._conf.num_measures)
        qualities = data.get("qualities") or []

        assert 0 <= instrument_id < self._conf.num_instruments, f"Instrument ID out of bounds {instrument_id}"
        assert 40 <= input_note_number <= 88, "Conditioning note number must be between" \
                                              " 40 and 88"
        assert 25 <= velocity <= 127, "Velocity must be between 25 and 127"
        assert np.shape(latent_sample) == (self._conf.latent_dim, ), f"Latent dim is wrong {np.shape(latent_sample)}"
        assert np.shape(heuristic_measures) == (self._conf.num_measures, )
        assert len(qualities) <= 18, f"Number of qualities can not be more than 18, found {len(qualities)}"

        measures_mean, note_index, velocity_index = self.get_measures_mean(
            input_pitch=input_note_number, velocity=velocity
        )

        if load_preset:
            # Find the appropriate instrument id based on qualities
            # This overrides the given measures and latent sample values
            logger.info("Selecting an instrument id based on input qualities")
            instrument_id = random.choice(self._find_iids(qualities))
            logger.info(f"Loading preset for instrument id {instrument_id}")
            success, values = self.load_preset_fn(measures_mean, note_index, velocity_index, instrument_id)
            if success:
                latent_sample, heuristic_measures = values

        processed_measures = [
            self.measure_transform(v, measures_mean[k])
            for k, v in zip(self._conf.data_handler.measure_names, heuristic_measures)
        ]

        mask = self._get_mask(output_note_number)

        decoder_inputs = {
            "mask": mask,
            "note_number": np.expand_dims(self._prepare_note_number(input_note_number), axis=0),
            "velocity": np.expand_dims(self._prepare_velocity(velocity), axis=0),
            "z": np.expand_dims(latent_sample, axis=0),
            "measures": np.expand_dims(np.array(processed_measures), axis=0),
            "measures_sliders": np.expand_dims(heuristic_measures, axis=0),
            "instrument_id": instrument_id
        }
        return decoder_inputs

    def get_prediction(self, data: Dict) -> Dict:
        output_dict = {
            "success": False,
            "audio": [],
            "measures_sliders": [0.5] * self._conf.num_measures,
            "z": [0.5] * self._conf.latent_dim
        }

        try:
            output_note_number = data.get("pitch") or 60
            decoder_inputs = self._prepare_inputs(data)
            logger.info("Getting prediction")
            prediction = self._model.decoder(decoder_inputs)
            logger.info("Transforming prediction")
            transformed = self._conf.data_handler.output_transform({}, prediction)
            logger.info("De-normalizing prediction")
            transformed["mask"] = decoder_inputs["mask"]
            transformed["note_number"] = output_note_number
            freq, mag, phase = self._conf.data_handler.denormalize(transformed)
            logger.info("Synthesising audio")
            audio = tsms.core.harmonic_synthesis(
                freq, mag, phase,
                self._conf.sample_rate,  self._conf.frame_size
            )
            output_dict.update({
                "success": True,
                "audio": np.squeeze(audio).tolist(),
                "measures_sliders": np.squeeze(decoder_inputs.get("measures_sliders")).tolist(),
                "z": np.squeeze(decoder_inputs.get("z")).tolist()
            })
            return output_dict
        except Exception as e:
            logger.error(e)
        return output_dict
