from typing import Dict


def get_config() -> Dict:
    conf = {
        "base_dir": "d:\soundofai\\nsynth-guitar-subset",
        "csv_file_path": "d:\soundofai\\annot_data\\updated.csv",
        "audio_duration": 4,
        "sample_rate": 16000,
        "num_classes": 9 * 2,
        "n_fft": 1024,
        "hop_len": 256,
        "n_mels": 256,
        "scale_factor": 1.0,
        "learning_rate": 3e-6,
        "threshold": 30
    }

    conf["time_steps"] = 1 + int(conf.get("audio_duration") * conf.get("sample_rate")/ conf.get("hop_len"))
    conf["num_conv_blocks"] = 5
    return conf
