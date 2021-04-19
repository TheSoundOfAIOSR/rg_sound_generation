import os
import click
import tqdm

from multiprocessing import Process
from pydub import AudioSegment


FORMATS = ['wav', 'mp3', 'ogg']


def convert_file(audio_file, from_format, to_format):
    audio = eval(f"AudioSegment.from_{from_format}('{audio_file}')")
    base_name = os.path.splitext(audio_file)[0]
    target_path = f'{base_name}.{to_format}'
    audio.export(target_path, format=to_format)


@click.command('convert-folder')
@click.option('--folder')
@click.option('--from_format', default='wav')
@click.option('--to_format', default='ogg')
def convert(folder: str, from_format: str, to_format: str) -> None:
    assert os.path.isdir(folder), f'{folder} must be a folder'
    assert from_format in FORMATS, f'formats must be of types: {FORMATS}'
    assert to_format in FORMATS, f'formats must be of types: {FORMATS}'

    audio_files = [os.path.join(folder, x) for x in os.listdir(folder) if x.lower().endswith(from_format)]
    remainder = len(audio_files) % 4
    audio_files += [None]*(4 - remainder)

    for f0, f1, f2, f3 in tqdm.tqdm(zip(*[iter(audio_files)] * 4)):
        p0, p1, p2, p3 = None, None, None, None
        if f0 is not None:
            p0 = Process(target=convert_file, args=(f0, from_format, to_format, ))
            p0.start()
        if f1 is not None:
            p1 = Process(target=convert_file, args=(f1, from_format, to_format, ))
            p1.start()
        if f2 is not None:
            p2 = Process(target=convert_file, args=(f2, from_format, to_format, ))
            p2.start()
        if f3 is not None:
            p3 = Process(target=convert_file, args=(f3, from_format, to_format, ))
            p3.start()

        if f0 is not None:
            p0.join()
        if f1 is not None:
            p1.join()
        if f2 is not None:
            p2.join()
        if f3 is not None:
            p3.join()


    click.echo('Conversion complete')


if __name__ == '__main__':
    convert()
