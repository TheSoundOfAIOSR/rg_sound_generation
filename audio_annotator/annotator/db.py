import os
import json
import random


file_path = os.path.join(os.getcwd(), 'annotator', 'static', 'examples.json')
audio_dir = os.path.join('static', 'audio')


def _read_data():
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data
    return {}


def _write_data(data):
    with open(file_path, 'w') as f:
        json.dump(data, f)


def get_random():
    data = _read_data()
    key = random.choice(list(data.keys()))
    value = data[key]
    qualities = value['qualities_str'] = [value['instrument_source_str']]
    audio_path = os.path.join(audio_dir, f'{key}.wav')
    return key, audio_path, qualities


def _validate_qualities(new_qualities):
    new_qualities = [q.lower() for q in new_qualities if q != '']
    return new_qualities


def add_qualities(key, new_qualities):
    data = _read_data()
    data[key]['qualities_str'] += _validate_qualities(new_qualities)
    _write_data(data)
