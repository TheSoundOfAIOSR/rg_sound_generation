import tensorflow as tf
import numpy as np
import glob
import soundfile as sf
import matplotlib.pyplot as plt
from . import tsms, core


def main():
    # file_list = glob.glob("samples/*.wav")
    # index = 1
    # audio_file = file_list[index]
    # note_number = int(audio_file[-11:-8])

    audio_file = "samples/synth_audio.wav"
    note_number = 50

    audio, sample_rate = sf.read(audio_file)

    audio = tf.cast(audio, dtype=tf.float32)
    audio = tf.reshape(audio, shape=(1, -1))

    frame_step = 64

    f0_estimate = tsms.core.midi_to_f0_estimate(
        note_number, audio.shape[1], frame_step)

    refined_f0_estimate, _, _ = tsms.core.refine_f0(
        audio, f0_estimate, sample_rate, frame_step)

    h_freq, h_mag, h_phase = tsms.core.iterative_harmonic_analysis(
        signals=audio,
        f0_estimate=refined_f0_estimate,
        sample_rate=sample_rate,
        frame_step=frame_step,
        corr_periods_list=[8.0] * 4,
        freq_smoothing_window=21)

    harmonic = tsms.core.harmonic_synthesis(
        h_freq, h_mag, h_phase, sample_rate, frame_step)
    harmonic = harmonic[:, :audio.shape[1]]

    residual = audio - harmonic

    # m0, m1 = core.harmonicity_measure(
    #     h_freq, h_mag, h_phase, residual, sample_rate, frame_step)

    # m0, m1 = core.even_odd_measure(
    #     h_freq, h_mag, h_phase, residual, sample_rate, frame_step)

    # m0, m1 = core.sparse_rich_measure(
    #     h_freq, h_mag, h_phase, residual, sample_rate, frame_step)

    # m0, m1 = core.vibrato_straight_measure(
    #     h_freq, h_mag, h_phase, residual, sample_rate, frame_step)

    # m0, m1 = core.hard_soft_attack_measure(
    #     h_freq, h_mag, h_phase, residual, sample_rate, frame_step)

    # print(m0.numpy(), m1.numpy())

    m = core.dark_measure(
        h_freq, h_mag, h_phase, residual, sample_rate, frame_step)

    print(m.numpy())


if __name__ == '__main__':
    main()
