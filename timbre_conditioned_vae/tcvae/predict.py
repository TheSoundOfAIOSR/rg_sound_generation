import tensorflow as tf
import numpy as np
import tsms
from . import model
from .train import get_inputs, get_all_measures
from .losses import deconstruct_tensors
from .compute_measures import heuristic_names
from .localconfig import LocalConfig


def load_model(conf: LocalConfig, checkpoint_path: str):
    _model = model.get_model_from_config(conf)
    _model.load_weights(checkpoint_path)
    return _model


def reconstruct_audio(freq, mag, conf):
    phase = tsms.core.generate_phase(
        freq, conf.sample_rate,
        conf.frame_size
    )
    return tsms.core.harmonic_synthesis(
        h_freq=freq, h_mag=mag, h_phase=phase,
        sample_rate=conf.sample_rate,
        frame_step=conf.frame_size
    )


def update_latent_input(note_number_orig, velocity_orig, heuristic_measures, **kwargs):
    for i, name in enumerate(heuristic_names):
        if name in kwargs:
            target_val = float(kwargs.get(name))
            print(f"Changing value for {name} from {heuristic_measures[0, i]} to {target_val}")
            heuristic_measures[0, i] = target_val
    if "note_number" in kwargs:
        target_val = kwargs.get("note_number")
        print(f"Changing note number from {note_number_orig} to {target_val}")
        note_number_orig = [target_val]
    if "velocity" in kwargs:
        target_index = int(kwargs.get("velocity"))
        print(f"Changing velocity from {np.argmax(velocity_orig)} to {target_index}")
        new_velocity = [0.] * 5
        new_velocity[target_index] = 1.
        velocity_orig = [new_velocity]
    return np.array(note_number_orig), np.array(velocity_orig), heuristic_measures


def get_intermediate_values(_model, batch, conf: LocalConfig, **kwargs):
    h, mask, note_number, velocity, _ = get_inputs(batch)
    note_number_orig = tf.argmax(note_number, axis=-1) + conf.starting_midi_pitch
    heuristic_measures = get_all_measures(batch, conf)

    note_number_orig, velocity, heuristic_measures = update_latent_input(
        note_number_orig.numpy(), velocity.numpy(),
        heuristic_measures.numpy(), **kwargs
    )
    if conf.use_encoder:
        if conf.is_variational:
            reconstruction, z_mean, z_log_var = _model.predict([h, note_number, velocity, heuristic_measures])
        else:
            reconstruction = _model.predict([h, note_number, velocity, heuristic_measures])
    else:
        reconstruction = _model.predict([note_number, velocity, heuristic_measures])
    return deconstruct_tensors(h, reconstruction, mask, conf), note_number_orig


def get_freq_and_mag_batch(_model, batch, conf: LocalConfig, **kwargs):
    (f0_shifts_true, f0_shifts_pred, mag_env_true, mag_env_pred,
     h_freq_shifts_true, h_freq_shifts_pred, h_mag_dist_true,
     h_mag_dist_pred, mask), note_number_orig = get_intermediate_values(_model, batch, conf, **kwargs)

    h_freq_true, h_mag_true = conf.data_handler.denormalize(
        f0_shifts_true, mag_env_true,
        h_freq_shifts_true, h_mag_dist_true,
        mask, note_number_orig, pred=False)

    h_freq_pred, h_mag_pred = conf.data_handler.denormalize(
        f0_shifts_pred, mag_env_pred,
        h_freq_shifts_pred, h_mag_dist_pred,
        mask, note_number_orig, pred=True)

    return h_freq_true, h_mag_true, h_freq_pred, h_mag_pred
