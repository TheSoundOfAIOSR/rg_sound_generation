import tensorflow as tf
import numpy as np
import os
import json
import pydub
import tsms
from tqdm import tqdm


audio_max_samples = 64000
min_pitch = 40
max_pitch = 88
sample_rate = 16000
frame_step = 64
valid_instrument_file = "maps/instruments_to_keep.csv"
instrument_map_file = "maps/instrument_id_map.csv"
instrument_to_index_file = "maps/instrument_to_index.json"
index_to_instrument_file = "maps/index_to_instrument.json"


def _fix_length(audio: np.ndarray, desired_num_samples: int):
    assert len(audio.shape) == 1

    num_samples = len(audio)
    if num_samples == desired_num_samples:
        return audio

    new_audio = np.zeros((desired_num_samples, ))

    if num_samples < desired_num_samples:
        new_audio[:num_samples] = audio
    else:
        new_audio = audio[:desired_num_samples]
    return new_audio.astype(np.float32)


def _load_audio(audio_path, sample_rate):
    with tf.io.gfile.GFile(audio_path, 'rb') as f:
        # Load audio at original SR
        audio_segment = (pydub.AudioSegment.from_file(f).set_channels(1))
        # Compute expected length at given `sample_rate`
        expected_len = int(audio_segment.duration_seconds * sample_rate)
        # Resample to `sample_rate`
        audio_segment = audio_segment.set_frame_rate(sample_rate)
        sample_arr = audio_segment.get_array_of_samples()
        audio = np.array(sample_arr).astype(np.float32)
        # Zero pad missing samples, if any
        audio = _fix_length(audio, audio_max_samples)
    # Convert from int to float representation.
    audio /= np.iinfo(sample_arr.typecode).max
    return audio


def _byte_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(
        float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(value=value))


def _tensor_feature(value):
    value = tf.constant(value)
    value = tf.io.serialize_tensor(value)
    value = tf.expand_dims(value, axis=0)
    return _byte_feature(value.numpy())


def _get_instrument_map():
    if not os.path.isfile(instrument_to_index_file) or not os.path.isfile(index_to_instrument_file):
        print("Instrument to index maps don't exist yet..")
        with open(valid_instrument_file, "r") as f:
            instruments = f.read().splitlines()
        with open(instrument_map_file, "r") as f:
            rows = f.read().splitlines()
            names = [name for name in rows[0].split(",") if name in instruments]
        instrument_to_index = dict((name, i) for i, name in enumerate(names))
        index_to_instrument = dict((value, key) for key, value in instrument_to_index.items())

        with open(instrument_to_index_file, "w") as f:
            json.dump(instrument_to_index, f)
        with open(index_to_instrument_file, "w") as f:
            json.dump(index_to_instrument, f)

    with open(instrument_to_index_file, "r") as f:
        instrument_to_index = json.load(f)

    return instrument_to_index


def prepare_example(sample_name, note_number,
                    velocity, instrument_id,
                    audio, h_freq, h_mag, h_phase):
    example_dict = {
        'sample_name': _byte_feature(sample_name),
        'note_number': _int64_feature(note_number),
        'velocity': _int64_feature(velocity),
        'instrument_id': _int64_feature(instrument_id),
        'audio': _float_feature(audio),
        'h_freq': _tensor_feature(h_freq),
        'h_mag': _tensor_feature(h_mag),
        'h_phase': _tensor_feature(h_phase)
    }
    return tf.train.Example(features=tf.train.Features(feature=example_dict))


def create_set(source_dir, target_dir, set_name="test"):
    assert os.path.isfile(valid_instrument_file)
    assert os.path.isfile(instrument_map_file)

    instrument_to_index = _get_instrument_map()

    set_dir = os.path.join(source_dir, set_name)
    audio_dir = os.path.join(set_dir, "audio")
    source_dataset_file = os.path.join(set_dir, "examples.json")
    target_dataset_file = os.path.join(target_dir, f"{set_name}.tfrecord")

    assert os.path.isdir(set_dir)
    assert os.path.isdir(audio_dir)
    assert os.path.isfile(source_dataset_file)

    with open(source_dataset_file, 'r') as file:
        source_dict = json.load(file)

    with tf.io.TFRecordWriter(target_dataset_file) as writer:
        for k, v in tqdm(source_dict.items()):
            instrument = str(k[7:-8])

            if instrument not in instrument_to_index:
                print(f"{instrument} not found in valid instruments, skipping..")
                continue

            instrument_id = instrument_to_index[instrument]
            note_number = v["pitch"]

            if note_number < min_pitch or note_number > max_pitch:
                print(f"{note_number} not found in valid range, skipping..")
                continue

            file_name = f"{k}.wav"
            target_path = os.path.join(audio_dir, file_name)

            audio = _load_audio(target_path, sample_rate)

            signals = tf.cast(audio, dtype=tf.float32)
            signals = tf.reshape(signals, shape=(1, -1))
            f0_estimate = tsms.core.midi_to_f0_estimate(note_number,
                                                        signals.shape[1],
                                                        frame_step)

            f0_estimate = tf.cast(f0_estimate, dtype=tf.float32)

            f0_estimate, _, _ = tsms.core.refine_f0(
                signals, f0_estimate, sample_rate, frame_step)

            h_freq, h_mag, h_phase = tsms.core.iterative_harmonic_analysis(
                signals=signals,
                f0_estimate=f0_estimate,
                sample_rate=sample_rate,
                frame_step=frame_step)

            h_freq = tf.squeeze(h_freq, axis=0)
            h_mag = tf.squeeze(h_mag, axis=0)
            h_phase = tf.squeeze(h_phase, axis=0)

            tf_example = prepare_example(sample_name=[str.encode(k)], note_number=[v['pitch']],
                                         velocity=[v['velocity']], instrument_id=[instrument_id],
                                         audio=audio, h_freq=h_freq, h_mag=h_mag, h_phase=h_phase)

            writer.write(tf_example.SerializeToString())
    print("Finished creating set")


def create_dataset(source_dir, target_dir):
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"No dir at {source_dir}")

    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)

    for set_name in ["test", "valid", "train"]:
        print("=" * 50)
        print(" " * 20, set_name, " " * 20)
        print("=" * 50)
        create_set(source_dir, target_dir, set_name=set_name)

    print(f"Finished creating dataset. {source_dir} can be removed to save space, if required")
