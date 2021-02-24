from dearpygui.core import *
from dearpygui.simple import *
from dialog import FolderDialog

import controller


class SettingsWindow:
    def __init__(self):
        with window(
            name='window_settings',
            label='Settings',
            x_pos=20,
            y_pos=20,
            width=300,
            height=100,
            no_close=True
        ):
            folder_dialog = FolderDialog(set_key='audio_folder_path')
            add_button(name='button_folder_dialog', label='Select Audio Folder', callback=folder_dialog.picker)
            add_text('', source='audio_folder_path')
            add_separator()
            add_button(name='button_load_folder', label='Load Files', callback=controller.load_files)
            add_same_line()
            add_button(name='button_show_logger', label='Show Logger', callback=controller.show_logger_window)
            add_same_line()
            add_button(name='button_exit', label='Exit', callback=controller.exit_application)


class FileBrowserWindow:
    def __init__(self):
        with window(
            name='window_file_browser',
            label='File Browser',
            x_pos=20,
            y_pos=140,
            width=300,
            height=300,
            no_close=True
        ):
            add_button(name='button_refresh_files', label='Refresh', callback=controller.load_files)
            add_same_line()
            add_button(name='button_next', label='Next', callback=controller.find_next_file)
            add_table(name='table_audio_files', headers=['File'], callback=controller.table_row_selected)
            add_separator()
            add_text('Selected file name: ')
            add_same_line()
            add_text('', source='selected_file_name')


class AudioPlayerWindow:
    def __init__(self):
        with window(
            name='window_audio_player',
            label='Audio PLayer',
            x_pos=340,
            y_pos=20,
            width=600,
            height=300,
            no_close=True
        ):
            add_button('button_play_audio', label='Play', callback=controller.play_audio)
            add_same_line()
            add_button('button_stop_audio', label='Stop', callback=controller.stop_audio)
            add_same_line()
            add_button('button_next_from_audio_window', label='Next', callback=controller.find_next_file)
            add_same_line()
            add_button('button_show_plot', label='Plot', callback=controller.get_audio_plot)
            add_separator()
            add_text('Selected file name: ')
            add_same_line()
            add_text('', source='selected_file_name')
            add_separator()
            # Audio sample plot
            add_plot("plot_sample", height=-1)


class AudioTagsWindow:
    def __init__(self):
        with window(
            name='window_audio_tags',
            label='Tags',
            x_pos=960,
            y_pos=20,
            no_close=True
        ):
            add_text('Enter relevant tags:')
            add_button(name='button_add_tag', label='+', callback=controller.add_new_tag_field)
            add_same_line()
            add_button(name='save_tags', label='Save', callback=controller.save_tags)
            add_separator()
