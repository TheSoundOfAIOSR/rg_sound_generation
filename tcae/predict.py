import tensorflow as tf
import numpy as np
from . import model
from .compute_measures import heuristic_names
from .localconfig import LocalConfig


def load_model(conf: LocalConfig, checkpoint_path: str):
    _model = model.get_model_from_config(conf)
    _model.load_weights(checkpoint_path)
    return _model


def update_latent_input(note_number, velocity, measures, **kwargs):
    for key in kwargs.keys():
        # ToDo Let Z be changed as well
        assert key in heuristic_names + ["velocity", "note_number"]
    for i, name in enumerate(heuristic_names):
        if name in kwargs:
            target_val = float(kwargs.get(name))
            print(f"Changing value for {name} from {measures[0, i]} to {target_val}")
            measures[0, i] = target_val
    if "note_number" in kwargs:
        target_val = kwargs.get("note_number")
        print(f"Changing note number from {note_number} to {target_val}")
        note_number = [target_val]
    if "velocity" in kwargs:
        target_index = int(kwargs.get("velocity"))
        print(f"Changing velocity from {np.argmax(velocity)} to {target_index}")
        new_velocity = [0.] * 5
        new_velocity[target_index] = 1.
        velocity = [new_velocity]
    return np.array(note_number), np.array(velocity), measures


def get_prediction(_model, batch, conf: LocalConfig, **kwargs):
    note_number_one_hot = batch["note_number"]
    velocity_one_hot = batch["velocity"]
    h = batch["h"]
    mask = batch["mask"]
    measures = batch["measures"]
    # note_number_one_hot, velocity_one_hot, h, mask, measures = batch

    note_number = tf.argmax(
        note_number_one_hot, axis=-1) + conf.starting_midi_pitch

    # note_number, velocity, measures = update_latent_input(
    #     note_number.numpy(), velocity_one_hot.numpy(),
    #     measures.numpy(), **kwargs
    # )

    model_input = [note_number_one_hot, velocity_one_hot]

    if conf.use_heuristics:
        model_input += [measures]
    if conf.use_encoder:
        model_input = [h] + model_input
    if conf.use_encoder and conf.is_variational:
        reconstruction, z_mean, z_log_var = _model.predict(model_input)
    else:
        reconstruction = _model.predict(model_input)

    h_true = h
    h_pred = reconstruction

    normalized_data_true = conf.data_handler.output_transform(
        h_true, pred=False)

    normalized_data_pred = conf.data_handler.output_transform(
        h_pred, pred=True)
    
    h_freq_true, h_mag_true, h_phase_true = conf.data_handler.denormalize(
        normalized_data_true, mask, note_number)

    h_freq_pred, h_mag_pred, h_phase_pred = conf.data_handler.denormalize(
        normalized_data_pred, mask, note_number)
    
    return (h_freq_true, h_mag_true, h_phase_true,
            h_freq_pred, h_mag_pred, h_phase_pred)
