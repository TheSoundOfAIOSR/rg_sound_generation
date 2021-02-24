"""
Annotation class encapsulates annotations for each audio file
"""
import json
import os


class Annotation:
    def __init__(self, file_path=None, audio_file_path=None):
        self.data = {
            'file_path': file_path or '',
            'audio_file_path': audio_file_path or '',
            'is_annotated': False,
            'tags': [],
            'texts': []
        }

    def load(self):
        file_path = self.data.get('file_path')
        if file_path != '':
            with open(file_path, 'r') as f:
                self.data = json.load(f)
        else:
            raise RuntimeError(f'file not found at {file_path}')

    def save(self):
        file_path = self.data.get('file_path')
        if file_path != '':
            with open(file_path, 'w') as f:
                json.dump(self.data, f)
        else:
            raise RuntimeError(f'file not found at {file_path}')

    def set_file_path(self, file_path):
        self.data['file_path'] = file_path

    def get_file_path(self):
        return self.data.get('file_path') or ''

    def set_audio_file_path(self, audio_file_path):
        self.data['audio_file_path'] = audio_file_path

    def get_audio_file_path(self):
        return self.data.get('audio_file_path') or ''

    def add_tag(self, tag):
        if tag not in self.data['tags']:
            self.data['tags'].append(tag)

    def add_tags(self, tags):
        self.data['tags'] = tags

    def add_text(self, text):
        text = text.lower()
        if text not in self.data['texts']:
            self.data['texts'].append(text)

    def get_tags(self):
        return self.data.get('tags')

    def get_texts(self):
        return self.data.get('texts')


class Data:
    def __init__(self, **kwargs):
        self.audio_folder_path = kwargs.get('audio_folder_path') or ''
        self.annotations = {
                'audio_files': [],
                'annotations': [],
                'is_annotated': []
            }

    @staticmethod
    def get_annotation_file_path(audio_file_path):
        return f'{audio_file_path[:-4]}.json'

    def load_annotations(self, audio_folder_path=None):
        if not audio_folder_path:
            audio_folder_path = self.audio_folder_path
        if not audio_folder_path:
            raise RuntimeError('No audio folder path found')
        self.annotations['audio_files'] = [x for x in os.listdir(audio_folder_path) if x.lower().endswith('.wav')]
        for f in self.annotations['audio_files']:
            audio_file_path = os.path.join(audio_folder_path, f)
            file_path = self.get_annotation_file_path(audio_file_path)
            annotation = Annotation(file_path=file_path, audio_file_path=audio_file_path)
            is_annotated = os.path.isfile(file_path) or annotation.data.get('is_annotated')
            if is_annotated:
                annotation.load()
            self.annotations['annotations'].append(annotation)
            self.annotations['is_annotated'].append(is_annotated)

    def get_annotations(self):
        return self.annotations

    def save_all(self):
        for annotation in self.annotations['annotations']:
            annotation.save()

    def save_annotation(self, index):
        self.annotations['annotations'][index].save()
        self.annotations['is_annotated'][index] = True

    def get_annotation(self, file_name):
        index = self.annotations['audio_files'].index(file_name)
        if index == -1:
            raise RuntimeError(f'No such file: {file_name}')
        return self.annotations['annotations'][index]

    def get_index(self, file_name):
        return self.annotations['audio_files'].index(file_name)


data = Data()
