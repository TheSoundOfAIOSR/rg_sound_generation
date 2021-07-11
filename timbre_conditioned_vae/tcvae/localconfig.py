import os
import json
from typing import Dict
from .utils import DataHandler


class LocalConfig:
    dataset_dir = os.path.join(os.getcwd(), "complete_dataset")
    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
    model_name = "VAE"
    run_name = "Default"
    best_model_path = None
    use_encoder = True
    latent_dim = 16
    hidden_dim = 256
    lstm_dim = 256
    lstm_dropout = 0.4
    harmonic_frame_steps = 1001
    frame_size = 64
    batch_size = 2
    num_instruments = 74
    starting_midi_pitch = 40
    num_pitches = 49
    num_velocities = 5
    max_num_harmonics = 98
    row_dim = 1024
    col_dim = 128
    padding = "same"
    epochs = 100
    early_stopping = 7
    learning_rate = 2e-4
    lr_plateau = 4
    lr_factor = 0.2
    gradient_norm = 5.
    csv_log_file = "logs.csv"
    final_conv_shape = (64, 8, 192) # update to be calculated dynamically
    final_conv_units = 64 * 8 * 192 # update to be calculated dynamically
    best_loss = 1e6
    sample_rate = 16000
    log_steps = True
    step_log_interval = 100
    kl_weight = 0.
    kl_weight_max = 1.
    kl_anneal_factor = 0.1
    kl_anneal_start = 10
    reconstruction_weight = 1.
    st_var = (2.0 ** (1.0 / 12.0) - 1.0)
    db_limit = -120
    decoder_type = "cnn"
    data_handler = DataHandler()

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalConfig, cls).__new__(cls)
        return cls._instance

    def set_config(self, params: Dict):
        vars(self).update(params)

    def load_config_from_file(self, file_path: str):
        assert os.path.isfile(file_path)

        with open(file_path, "r") as f:
            params = json.load(f)
        self.set_config(params)

    def save_config(self):
        target_path = os.path.join(self.checkpoints_dir, f"{self.run_name}_{self.model_name}.json")

        with open(target_path, "w") as f:
            json.dump(vars(self), f)
