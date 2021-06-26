import os


class LocalConfig:
    dataset_dir = os.path.join(os.getcwd(), "complete_dataset")
    checkpoints_dir = os.path.join(os.getcwd(), "checkpoints")
    model_name = "VAE"
    latent_dim = 1
    hidden_dim = 256
    harmonic_frame_steps = 1001
    frame_size = 64
    batch_size = 2
    num_instruments = 74
    starting_midi_pitch = 40
    num_pitches = 49
    num_velocities = 5
    max_num_harmonics = 99
    row_dim = 1024
    col_dim = 128
    padding = "same"
    epochs = 100
    early_stopping = 7
    learning_rate = 2e-5
    gradient_norm = 5.
    csv_log_file = "logs.csv"
    final_conv_shape = (16, 2, 288) # update to be calculated dynamically
    final_conv_units = 16 * 2 * 288 # update to be calculated dynamically
    best_loss = 1e6
    sample_rate = 16000
    log_steps = True
    step_log_interval = 100
    decoder_type = "conv" # "mlp"
    kl_weight = 0.
    kl_anneal_factor = 0.1
    freq_loss_weight = 1000.
    st_var = (2.0 ** (1.0 / 12.0) - 1.0)
    db_limit = -120

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalConfig, cls).__new__(cls)
        return cls._instance
