"""
Audio Annotator
Author: https://github.com/am1tyadav
"""

from dearpygui.core import *

import ui


if __name__ == '__main__':
    set_main_window_title('Audio Annotator')
    set_logger_window_title('Logs')
    # UI
    ui.SettingsWindow()
    ui.FileBrowserWindow()
    ui.AudioPlayerWindow()
    ui.AudioTagsWindow()
    show_logger()
    # Run the app
    set_theme('Dark 2')
    start_dearpygui()
