import os
import shutil
import json
import wget
import tarfile
from tqdm import tqdm


def _download_nsynth(download_dir):
    train = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz"
    valid = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz"
    test = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"

    if not os.path.isdir(download_dir):
        os.mkdir(download_dir)

    for url in [test, valid, train]:
        base_name = os.path.basename(url)
        print(f"Downloading {base_name}")
        if not os.path.isfile(os.path.join(download_dir, base_name)):
            wget.download(url, out=download_dir)
        else:
            print("Already downloaded")


def _extract_nsynth(download_dir):
    paths = []
    for set_name in ["train", "valid", "test"]:
        base_name = f"nsynth-{set_name}"
        p = os.path.join(download_dir, base_name)
        if not os.path.isdir(p):
            paths.append(f"{p}.jsonwav.tar.gz")
        else:
            print(f"{p} is already extracted")

    for p in paths:
        print(f"Extracting {p}")
        tar = tarfile.open(p)
        tar.extractall(download_dir)
        tar.close()


def _copy_audio(filename, source_dir, target_dir):
    filepath = os.path.join(source_dir, f'{filename}.wav')
    target_path = os.path.join(target_dir, "audio", f'{filename}.wav')
    
    assert os.path.isfile(filepath), f'File {filepath} does not exist'
    assert os.path.isdir(target_dir), f'Folder {target_dir} does not exist'
    
    shutil.copy(filepath, target_path)


def _create_set(setname, source_dir, target_dir, instrument):
    print(f'Creating subset for instrument id {instrument} for {setname} set')
    
    assert setname in ['train', 'valid', 'test'], 'Not a valid set name'
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
        
    target_set_dir = os.path.join(target_dir, setname)
    source_set_dir = os.path.join(source_dir, f'nsynth-{setname}/audio')
    
    assert os.path.isdir(source_set_dir), 'Source audio folder not found'
    
    if not os.path.isdir(target_set_dir):
        os.mkdir(target_set_dir)
        os.mkdir(os.path.join(target_set_dir, 'audio'))
    
    with open(os.path.join(source_dir, f'nsynth-{setname}/examples.json'), 'r') as f:
        dataset = json.load(f)
    
    print(f'Subset will be created at {target_set_dir}')
    
    subset = {}
    
    for k, v in tqdm(dataset.items()):
        if v.get('instrument_family') == instrument:
            subset[k] = v
            _copy_audio(k, source_set_dir, target_set_dir)
    
    with open(os.path.join(target_set_dir, 'examples.json'), 'w') as f:
        json.dump(subset, f)


def create_subset(source_dir, target_dir, instrument=3):
    _download_nsynth(source_dir)
    _extract_nsynth(source_dir)

    for set_name in ["test", "valid", "train"]:
        print("="*50)
        print(" "*20, set_name, " "*20)
        print("="*50)
        _create_set(set_name, source_dir, target_dir, instrument)

    print(f"You can now delete {source_dir} if you do not plan on using the complete NSynth dataset")
