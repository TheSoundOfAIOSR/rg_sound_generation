import os
import json
from typing import Dict
from .data_handler import DataHandler, SimpleDataHandler


class LocalConfig:
    dataset_dir = os.path.join(os.getcwd(), "complete_dataset")
    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
    simple_encoder = False
    simple_decoder = False
    pretrained_model_path = None
    model_name = "VAE"
    run_name = "Default"
    best_model_path = None
    use_encoder = True
    use_phase = False
    latent_dim = 16
    use_max_pool = True
    strides = 2
    use_lstm_in_encoder = True
    use_note_number = True
    use_velocity = True
    use_heuristics = True
    hidden_dim = 256
    default_k = 3
    deep_decoder = False
    add_z_to_decoder_blocks = True
    check_decoder_hidden_dim = True
    print_model_summary = False
    skip_channels = 32
    lstm_dim = 256
    lstm_dropout = 0.4
    harmonic_frame_steps = 1001
    max_harmonics = 110
    frame_size = 64
    batch_size = 2
    num_instruments = 74
    num_measures = 7 + 4
    starting_midi_pitch = 40
    num_pitches = 49
    num_velocities = 5
    max_num_harmonics = 98
    row_dim = 1024
    col_dim = 128
    padding = "same"
    epochs = 500
    num_train_steps = None
    num_valid_steps = None
    early_stopping = 7
    learning_rate = 2e-4
    lr_plateau = 4
    lr_factor = 0.5
    gradient_norm = 5.
    csv_log_file = "logs.csv"
    final_conv_shape = (64, 8, 192)  # ToDo: to be calculated dynamically
    final_conv_units = 64 * 8 * 192  # ToDo: to be calculated dynamically
    best_loss = 1e6
    sample_rate = 16000
    log_steps = True
    step_log_interval = 100
    is_variational = True
    using_mt = True

    mt_inputs = {
        "f0_shifts": {"shape": (row_dim, 32)},
        "h_freq_shifts": {"shape": (row_dim, 64)},
        "mag_env": {"shape": (row_dim, 32)},
        "h_mag_dist": {"shape": (row_dim, 64)},
        "h_phase_diff": {"shape": (row_dim, 64)},
    }
    mt_outputs = {
        "f0_shifts": {"enabled": True, "shape": (row_dim, 64, 16), "channels": 1},
        "h_freq_shifts": {"enabled": True, "shape": (row_dim, 110, 16), "channels": 110},
        "mag_env": {"enabled": True, "shape": (row_dim, 64, 16), "channels": 1},
        "h_mag_dist": {"enabled": True, "shape": (row_dim, 110, 16), "channels": 110},
        "h_phase_diff": {"enabled": False, "shape": (row_dim, 110, 16), "channels": 110},
    }

    use_kl_anneal = False
    kl_weight = 1.
    kl_weight_max = 1.
    kl_anneal_factor = 0.05
    kl_anneal_start = 20
    reconstruction_weight = 1.
    st_var = (2.0 ** (1.0 / 12.0) - 1.0)
    db_limit = -120
    encoder_type = "2d" # or "1d"
    decoder_type = "cnn"
    freq_bands = {
        "bass": [60, 270],
        "mid": [270, 2000],
        "high_mid": [2000, 6000],
        "high": [6000, 20000]
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

    def set_data_handler_by_type(self, data_handler_type: str):
        assert data_handler_type in ["data_handler", "simple_data_handler"]
        self.data_handler_type = data_handler_type
        if data_handler_type == "data_handler":
            self.data_handler = DataHandler()
            self.data_handler_properties = [
                "use_phase",
                "weight_type",
                "mag_loss_type",
                "mag_scale_fn"
            ]
        elif data_handler_type == "simple_data_handler":
            self.data_handler = SimpleDataHandler()

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
            rows = f.read().splitlines()
        for r in rows:
            if len(r) == 0:
                continue
            if r.startswith("#"):
                continue
            if r.startswith("conf."):
                command = f"self.{r[5:]}"
                if command.startswith("self.save_config"):
                    continue
                exec(command)

    def save_config(self):
        target_path = os.path.join(self.checkpoints_dir,
                                   f"{self.run_name}_{self.model_name}.json")

        to_save = vars(self).copy()

        for p in self.data_handler_properties:
            to_save[p] = eval(f"self.data_handler.{p}")

        if "data_handler" in to_save:
            # to_save is not a deep copy so, we pop item
            to_save.pop("data_handler")

        with open(target_path, "w") as f:
            json.dump(to_save, f)
