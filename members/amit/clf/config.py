from typing import Dict

# "base_dir": "D:\soundofai\\pitch_shifted_all",

def get_config() -> Dict:
    conf = {
        "base_dir": "D:\soundofai\\all_nsynth_audio",
        "csv_file_path": "D:\soundofai\\annot_data\\data\\june_06.csv",
        "fb_categorical_file": "..\\fb_categorical.json",
        "preprocess_dir": "tmp",
        "audio_duration": 4,
        "clip_at": -30,
        "epsilon": 1e-5,
        "clip_audio_at": 2,
        "sample_rate": 16000,
        "num_classes": 12,
        "n_fft": 2048,
        "hop_len": 512,
        "n_mels": 128,
        "scale_factor": 1.0,
        "learning_rate": 2e-4,
        "threshold": 15,
        "all_features": [
            'bright_vs_dark', 'full_vs_hollow', 'smooth_vs_rough',
            'warm_vs_metallic', 'clear_vs_muddy', 'thin_vs_thick',
            'pure_vs_noisy', 'rich_vs_sparse', 'soft_vs_hard'
        ],
        "features": ["bright_vs_dark"],
        "model_name": "fb_qualities",
        "valid_split": 0.4,
        "dry_run": False,
        "reset_data": False,
        "pitch_shifted": True
    }

    conf = dict(conf)
    audio_duration_samples = (conf.get("audio_duration") - conf.get("clip_audio_at")) * conf.get("sample_rate")
    conf["time_steps"] = 1 + audio_duration_samples // conf.get("hop_len")
    conf["num_conv_blocks"] = 3
    return conf
