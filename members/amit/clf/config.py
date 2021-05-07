from typing import Dict

# "csv_file_path": "D:\soundofai\\annot_data\\data\\may_05.csv",

def get_config() -> Dict:
    conf = {
        "base_dir": "D:\soundofai\\pitch_shifted_all",
        "csv_file_path": "D:\soundofai\\annot_data\\data\\lucas_07_05.csv",
        "preprocess_dir": "tmp",
        "audio_duration": 4,
        "sample_rate": 16000,
        "num_classes": 3,
        "n_fft": 1024,
        "hop_len": 512,
        "n_mels": 128,
        "scale_factor": 0.2,
        "learning_rate": 3e-6,
        "threshold": 10,
        "all_features": [
            'bright_vs_dark', 'full_vs_hollow', 'smooth_vs_rough',
            'warm_vs_metallic', 'clear_vs_muddy', 'thin_vs_thick',
            'pure_vs_noisy', 'rich_vs_sparse', 'soft_vs_hard'
        ],
        "features": ["bright_vs_dark"],
        "model_name": "bright_vs_dark",
        "valid_split": 0.2
    }

    conf = dict(conf)
    conf["time_steps"] = 1 + int(conf.get("audio_duration") * conf.get("sample_rate")/ conf.get("hop_len"))
    conf["num_conv_blocks"] = 5
    return conf
