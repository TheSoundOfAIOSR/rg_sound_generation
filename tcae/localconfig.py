import os
import json
from typing import Dict
from tcae.data_handler import DataHandler


class LocalConfig:
    dataset_dir = os.path.join(os.getcwd(), "complete_dataset")
    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
    dataset_modifier = None
    simple_encoder = False
    simple_decoder = False
    use_fnet_skip_dense = True
    create_encoder_function = 'mt'
    create_decoder_function = 'cnn'
    lc_dropout_rate = 0.1
    pretrained_model_path = None
    model_name = "tcae"
    run_name = "default"
    best_model_path = None
    use_encoder = True
    use_phase = False
    latent_dim = 16
    # strides = 2
    use_note_number = True
    use_velocity = True
    use_instrument_id = False
    use_heuristics = True
    use_one_hot_conditioning = True
    hidden_dim = 256
    print_model_summary = False
    lstm_dim = 256
    lstm_dropout = 0.4
    batch_size = 2
    num_instruments = 74
    num_measures = 7 + 4
    starting_midi_pitch = 40
    num_pitches = 49
    num_velocities = 5
    max_num_harmonics = 110
    frame_size = 64
    harmonic_frame_steps = 1000
    row_dim = 1024
    padding = "same"
    epochs = 500
    early_stopping = 7
    learning_rate = 3e-4
    lr_plateau = 4
    lr_factor = 0.5
    csv_log_file = "logs.csv"
    sample_rate = 16000
    using_categorical = False
    use_embeddings = False
    scalar_embedding = False
    pitch_emb_size = 16
    velocity_emb_size = 4

    mt_inputs = {
        "f0_shifts": {"shape": (row_dim, 32)},
        "h_freq_shifts": {"shape": (row_dim, 64)},
        "mag_env": {"shape": (row_dim, 32)},
        "h_mag_dist": {"shape": (row_dim, 64)},
        "h_freq_correction": {"shape": (row_dim, 64)},
    }

    mt_outputs = {
        "f0_shifts": {"shape": (row_dim, 64, 16)},
        "h_freq_shifts": {"shape": (row_dim, 110, 16)},
        "mag_env": {"shape": (row_dim, 64, 16)},
        "h_mag_dist": {"shape": (row_dim, 110, 16)},
        "h_freq_correction": {"shape": (row_dim, 110, 16)},
    }

    data_handler = None
    data_handler_properties = []
    data_handler_type = "none"

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LocalConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_handler_type="data_handler"):
        if self.data_handler is None:
            self.set_data_handler_by_type(data_handler_type)
        self.data_handler.starting_midi_pitch = self.starting_midi_pitch

    def set_data_handler_by_type(self, data_handler_type: str):
        assert data_handler_type in ["data_handler", "simple_data_handler"]
        self.data_handler_type = data_handler_type
        if data_handler_type == "data_handler":
            self.data_handler = DataHandler()
            self.data_handler_properties = [
                "weight_type",
                "mag_loss_type",
                "mag_scale_fn",
                "freq_loss_type"
            ]

    def set_config(self, params: Dict):
        params_conf = dict((k, v) for k, v in params.items()
                           if k not in self.data_handler_properties)

        vars(self).update(params_conf)

        for p in self.data_handler_properties:
            if p in params:
                exec(f"self.data_handler.{p} = params['{p}']")

    def load_config_from_file(self, file_path: str):
        assert os.path.isfile(file_path)
        file_ext = os.path.splitext(file_path)[-1]
        assert file_ext in [".json", ".txt"]
        if file_ext == ".json":
            self.load_config_from_json(file_path)
        elif file_ext == ".txt":
            self.load_config_from_text(file_path)

    def load_config_from_json(self, file_path: str):
        with open(file_path, "r") as f:
            params = json.load(f)

        if "data_handler_type" in params:
            self.set_data_handler_by_type(params["data_handler_type"])

        self.set_config(params)

    def load_config_from_text(self, file_path: str):
        assert self.data_handler is not None
        with open(file_path, "r") as f:
            file_data = f.read()
        file_data = file_data.replace("conf.", "self.")
        file_data = file_data.replace("self.save_config", "# ")
        exec(file_data)

    def save_config(self):
        target_path = os.path.join(self.checkpoints_dir,
                                   f"{self.run_name}_{self.model_name}.json")

        to_save = vars(self).copy()

        for p in self.data_handler_properties:
            to_save[p] = eval(f"self.data_handler.{p}")

        if "data_handler" in to_save:
            # to_save will not save objects, we pop item
            to_save.pop("data_handler")

        with open(target_path, "w") as f:
            json.dump(to_save, f)
