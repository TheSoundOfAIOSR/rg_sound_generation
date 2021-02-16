"""
Creates a subset of an instrument from NSynth dataset
Run after download NSynth dataset from https://magenta.tensorflow.org/datasets/nsynth
Run this script in the same folder that you extract Nsynth dataset ^

If you have not downloaded NSynth dataset yet, run the following commands in terminal
in a directory that you want to download the dataset in:

wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz
tar -xf nsynth-train.jsonwav.tar.gz
tar -xf nsynth-valid.jsonwav.tar.gz
tar -xf nsynth-test.jsonwav.tar.gz
"""

import os
import shutil
import json
import click

from tqdm import tqdm


def _copy_audio(filename, source_dir, target_dir):
    filepath = os.path.join(source_dir, f'{filename}.wav')
    target_path = os.path.join(target_dir, 'audio', f'{filename}.wav')
    
    assert os.path.isfile(filepath), f'File {filepath} does not exist'
    assert os.path.isdir(target_dir), f'Folder {target_dir} does not exist'
    
    shutil.copy(filepath, target_path)


@click.command()
@click.option('--setname', help='One of train, valid or test')
@click.option('--root_dir', help='Root directory where sets are to be created')
@click.option('--instrument', default=3, help='Id of instrument to target')
def create_subset(setname, root_dir, instrument=3):
    """
    Creates an instrument subset from nsynth dataset
    Arguments:
        setname: can be either 'train', 'valid' or 'test'
        root_dir: root directory of where the sets are to be created
        instrument: id of the instrument to target. default=3 for guitar
    """
    print(f'Creating subset for instrument id {instrument} for {setname} set')
    
    assert setname in ['train', 'valid', 'test'], 'Not a valid setname'
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)
        
    target_dir = os.path.join(root_dir, setname)
    source_dir = f'./nsynth-{setname}/audio'
    
    assert os.path.isdir(source_dir), 'Source audio folder not found'
    
    if not os.path.isdir(target_dir):
        os.mkdir(target_dir)
        os.mkdir(os.path.join(target_dir, 'audio'))
    
    with open(f'./nsynth-{setname}/examples.json', 'r') as f:
        dataset = json.load(f)
    
    print(f'Subset will be created at {target_dir}')
    
    subset = {}
    
    for k, v in tqdm(dataset.items()):
        if v.get('instrument_family') == instrument:
            subset[k] = v
            _copy_audio(k, source_dir, target_dir)
    
    with open(os.path.join(target_dir, 'examples.json'), 'w') as f:
        json.dump(subset, f)

        
if __name__ == '__main__':
    create_subset()
