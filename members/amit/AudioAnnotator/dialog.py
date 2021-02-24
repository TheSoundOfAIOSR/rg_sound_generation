from dearpygui.core import *

import os


class FolderDialog:
    def __init__(self, **kwargs):
        super(FolderDialog, self).__init__()
        self.set_key = kwargs.get('set_key')

    def picker(self, sender, data):
        select_directory_dialog(callback=self._apply_selected)

    def _apply_selected(self, sender, data):
        base_path, folder_name = data
        folder_path = os.path.join(base_path, folder_name)
        set_value(self.set_key, folder_path)


class FileDialog:
    def __init__(self, **kwargs):
        super(FileDialog, self).__init__()
        self.set_key = kwargs.get('set_key')

    def picker(self, sender, data):
        open_file_dialog(callback=self._apply_selected)

    def _apply_selected(self, sender, data):
        base_path, file_name = data
        file_path = os.path.join(base_path, file_name)
        set_value(self.set_key, file_path)
