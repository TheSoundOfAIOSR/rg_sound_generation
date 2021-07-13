import tensorflow as tf
import tsms
from . import model
from .train import get_inputs, get_all_measures
from .losses import deconstruct_tensors
from .localconfig import LocalConfig


def load_model(conf: LocalConfig, checkpoint_path: str):
    decoder = model.create_rnn_decoder(conf) if conf.decoder_type == "rnn" else model.create_decoder(conf)
    decoder.load_weights(checkpoint_path)
    return decoder


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


def get_freq_and_mag(f0_shifts, mag_env, h_freq_shifts,
                     h_mag_distribution, mask, note_number_orig, conf):
    h_freq, h_mag = conf.data_handler.denormalize(f0_shifts, mag_env,
                                                  h_freq_shifts, h_mag_distribution,
                                                  mask, note_number_orig)
    # h_freq = tf.squeeze(h_freq, axis=-1)
    # h_mag = tf.squeeze(h_mag, axis=-1)

    h_freq_reshaped, h_mag_reshaped = [], []

    for i in range(0, conf.batch_size):
        freq, mag = conf.data_handler.reshape_freq_mag(
            tf.expand_dims(h_freq[i, ...], axis=0),
            tf.expand_dims(h_mag[i, ...], axis=0)
        )
        h_freq_reshaped.append(freq)
        h_mag_reshaped.append(mag)
    return h_freq_reshaped, h_mag_reshaped


def get_freq_and_mag_batch(decoder, batch, conf: LocalConfig, **kwargs):
    h, mask, note_number, velocity, _ = get_inputs(batch)
    note_number_orig = tf.argmax(note_number, axis=-1) + conf.starting_midi_pitch
    heuristic_measures = get_all_measures(batch, conf)

    # ToDo: Change note, velocity, heuristic measures based on kwargs
    reconstruction = decoder.predict([note_number, velocity, heuristic_measures])

    (f0_shifts_true, f0_shifts_pred, mag_env_true, mag_env_pred,
     h_freq_shifts_true, h_freq_shifts_pred, h_mag_distribution_true,
     h_mag_distribution_pred, mask) = deconstruct_tensors(h, reconstruction, mask, conf)

    h_freq_true_reshaped, h_mag_true_reshaped = get_freq_and_mag(
        f0_shifts_true, mag_env_true, h_freq_shifts_true,
        h_mag_distribution_true, mask, note_number_orig, conf
    )

    h_freq_pred_reshaped, h_mag_pred_reshaped = get_freq_and_mag(
        f0_shifts_pred, mag_env_pred, h_freq_shifts_pred,
        h_mag_distribution_pred, mask, note_number_orig, conf
    )
    return (h_freq_true_reshaped, h_mag_true_reshaped,
            h_freq_pred_reshaped, h_mag_pred_reshaped)


def dummy(decoder, batch, conf: LocalConfig, **kwargs):
    h, mask, note_number, velocity, _ = get_inputs(batch)
    note_number_orig = tf.argmax(note_number, axis=-1) + conf.starting_midi_pitch
    heuristic_measures = get_all_measures(batch, conf)

    # ToDo: Change note, velocity, heuristic measures based on kwargs
    reconstruction = decoder.predict([note_number, velocity, heuristic_measures])

    return deconstruct_tensors(h, reconstruction, mask, conf), note_number_orig
