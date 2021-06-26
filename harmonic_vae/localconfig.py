class LocalConfig:
    dataset_dir = "D:\soundofai\\cleaned_nsynth"
    latent_dim = 128
    hidden_dim = 256
    harmonic_frame_steps = 1001
    frame_size = 64
    batch_size = 2
    num_instruments = 74
    num_pitches = 49
    num_velocities = 5
    max_num_harmonics = 99
    row_dim = 1024
    col_dim = 128
    padding = "same"
    epochs = 100
    early_stopping = 7
    learning_rate = 2e-8
    gradient_norm = 5.
    csv_log_file = "logs.csv"
    final_conv_shape = (16, 2, 288)
    final_conv_units = 16 * 2 * 288
    best_loss = 1e6
    sample_rate = 16000
    log_steps = True
    step_log_interval = 100
    decoder_type = "conv" # "mlp"
    kl_weight = 0.1

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalConfig, cls).__new__(cls)
        return cls._instance
