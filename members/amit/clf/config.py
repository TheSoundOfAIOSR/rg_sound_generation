from typing import Dict


def get_config() -> Dict:
    conf = {
        "base_dir": "D:\soundofai\\pitch_shifted_all",
        "csv_file_path": "D:\soundofai\\annot_data\\data\\may_05.csv",
        "audio_duration": 4,
        "sample_rate": 16000,
        "num_classes": 9,
        "n_fft": 1024,
        "hop_len": 512,
        "n_mels": 256,
        "scale_factor": 0.2,
        "learning_rate": 3e-6,
        "threshold": 25,
        "all_features": [
            'bright_vs_dark', 'full_vs_hollow', 'smooth_vs_rough',
            'warm_vs_metallic', 'clear_vs_muddy', 'thin_vs_thick',
            'pure_vs_noisy', 'rich_vs_sparse', 'soft_vs_hard'
        ],
        "features": ["bright_vs_dark"],
        "model_name": "bright_vs_dark"
    }

    conf["time_steps"] = 1 + int(conf.get("audio_duration") * conf.get("sample_rate")/ conf.get("hop_len"))
    conf["num_conv_blocks"] = 5
    conf["num_classes"] = len(conf["features"]) * 2
    return conf
