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
    note_number_one_hot, velocity_one_hot, h, mask, measures = batch

    note_number = tf.argmax(
        note_number_one_hot, axis=-1) + conf.starting_midi_pitch

    # note_number, velocity, measures = update_latent_input(
    #     note_number.numpy(), velocity_one_hot.numpy(),
    #     measures.numpy(), **kwargs
    # )

    if conf.use_encoder:
        if conf.is_variational:
            reconstruction, z_mean, z_log_var = _model.predict(
                [h, note_number_one_hot, velocity_one_hot, measures])
        else:
            reconstruction = _model.predict(
                [h, note_number_one_hot, velocity_one_hot, measures])
    else:
        reconstruction = _model.predict(
            [note_number_one_hot, velocity_one_hot, measures])

    f0_shifts_true, mag_env_true, \
    h_freq_shifts_true, h_mag_dist_true, h_phase_diff_true = \
        conf.data_handler.output_transform(h, pred=False)

    f0_shifts_pred, mag_env_pred, \
    h_freq_shifts_pred, h_mag_dist_pred, h_phase_diff_pred = \
        conf.data_handler.output_transform(reconstruction, pred=True)
    
    h_freq_true, h_mag_true, h_phase_true = conf.data_handler.denormalize(
        f0_shifts_true, mag_env_true,
        h_freq_shifts_true, h_mag_dist_true,
        h_phase_diff_true,
        mask, note_number, pred=False)

    h_freq_pred, h_mag_pred, h_phase_pred = conf.data_handler.denormalize(
        f0_shifts_pred, mag_env_pred,
        h_freq_shifts_pred, h_mag_dist_pred,
        h_phase_diff_pred,
        mask, note_number, pred=True)
    
    return h_freq_true, h_mag_true, h_phase_true, \
           h_freq_pred, h_mag_pred, h_phase_pred
