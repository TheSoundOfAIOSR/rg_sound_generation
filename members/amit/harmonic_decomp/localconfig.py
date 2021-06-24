class LocalConfig:
    dataset_dir = "D:\soundofai\\cleaned_nsynth"
    latent_dim = 64
    hidden_dim = 128
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
    gradient_norm = 2.
    csv_log_file = "logs.csv"
    final_conv_shape = (16, 2, 288)
    final_conv_units = 16 * 2 * 288

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LocalConfig, cls).__new__(cls)
        return cls._instance
