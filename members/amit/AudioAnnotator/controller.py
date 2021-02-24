from dearpygui.core import *
from scipy.io import wavfile

import sys
import ui
import annotations
import simpleaudio


current_tags = []


def load_files(_):
    try:
        audio_folder_path = get_value('audio_folder_path')
        annotations.data.load_annotations(audio_folder_path)
        audio_files = annotations.data.get_annotations().get('audio_files')
        clear_table('table_audio_files')
        for f in audio_files:
            add_row('table_audio_files', [f])
        log_info('Files loaded')
    except Exception as e:
        log_error(e)


def set_current_audio_file(index):
    set_value('selected_file_name', annotations.data.get_annotations().get('audio_files')[index])
    if index != -1:
        get_audio_plot(None)
        # Reset stuff
        global current_tags
        delete_item('window_audio_tags')
        current_tags = get_annotation().get_tags() or []
        ui.AudioTagsWindow()
        if bool(current_tags):
            for i, tag in enumerate(current_tags):
                add_input_text(f'tag_{i}', parent='window_audio_tags', default_value=tag)


def find_next_file(_):
    is_annotated_list = annotations.data.get_annotations().get('is_annotated')
    if False in is_annotated_list:
        index = is_annotated_list.index(False)
        set_current_audio_file(index)
    else:
        log_info('No more unsaved annotations')


def show_logger_window(_):
    show_logger()


def exit_application(_):
    sys.exit(0)


def get_audio_file_path():
    file_name = get_value('selected_file_name')
    return annotations.data.get_annotation(file_name).get_audio_file_path()


def get_annotation():
    try:
        file_name = get_value('selected_file_name')
        return annotations.data.get_annotation(file_name)
    except Exception as e:
        log_error(e)
    return None


def get_audio_plot(_):
    try:
        _, audio = wavfile.read(get_audio_file_path())
        time_axis = list(range(0, audio.shape[0]))
        amplitude = [float(x) for x in audio]
        clear_plot('plot_sample')
        add_line_series("plot_sample", "Amplitude", time_axis, amplitude, weight=4, color=[255, 0, 0, 100])
    except Exception as e:
        log_error(e)


def play_audio(_):
    try:
        stop_audio(_)
        log_info('Playing audio')
        simpleaudio.WaveObject.from_wave_file(get_audio_file_path()).play()
    except Exception as e:
        log_error(e)


def stop_audio(_):
    try:
        log_info('Stopping..')
        simpleaudio.stop_all()
    except Exception as e:
        log_error(e)


def table_row_selected(_):
    selected_row_items = get_table_selections('table_audio_files')
    if len(selected_row_items) > 0:
        selected_row_items = selected_row_items[-1]
    selected_index, _ = selected_row_items
    set_table_selection('table_audio_files', selected_index, 0, False)
    set_current_audio_file(selected_index)


def add_new_tag_field(_):
    try:
        add_input_text(f'tag_{len(current_tags)}', parent='window_audio_tags')
        current_tags.append('')
    except Exception as e:
        log_error(e)


def save_tags(_):
    try:
        global current_tags
        # Save stuff
        tags = [get_value(f'tag_{i}') for i, _ in enumerate(current_tags)]
        tags = [x for x in list(set(tags)) if x != '']
        annot = get_annotation()
        annot.add_tags(tags)
        index = annotations.data.get_index(get_value('selected_file_name'))
        annotations.data.save_annotation(index)
        log_info('Saved annotation')
    except Exception as e:
        log_error(e)
