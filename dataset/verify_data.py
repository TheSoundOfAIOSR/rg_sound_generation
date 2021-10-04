# Load and verify data against original example.json

import os
import json
from prepare import CompleteTFRecordProvider


base_dir = "D:\soundofai\\nsynth-guitar-subset"
sets = ["test", "valid", "train"]

with open("maps/instrument_to_index.json", "r") as f:
    instrument_to_index = json.load(f)

for s in sets:
    print("=" * 50)
    print(f"Verifying {s}")
    print("=" * 50)

    set_path = os.path.join(base_dir, s)
    dp = CompleteTFRecordProvider(file_pattern=os.path.join(set_path, "complete.tfrecord"))
    dataset = dp.get_batch(1, shuffle=False, repeats=1)

    with open(os.path.join(set_path, "examples.json"), "r") as f:
        original_data = json.load(f)

    for e in dataset:
        sample_name = e["sample_name"].numpy()[0][0].decode()
        instrument = sample_name[7:-8]
        instrument_id = instrument_to_index[instrument]

        id_in_data = e["instrument_id"].numpy()[0][0]
        note_number = e['note_number'].numpy()[0][0]

        value = original_data[sample_name]

        print(f"Verifying {sample_name}")
        assert note_number == value["pitch"]
        assert instrument_id == id_in_data
