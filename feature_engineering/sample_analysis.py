import os
import tensorflow as tf
import numpy as np
import sound_models as sm
import ddsp
import soundfile as sf
import matplotlib.pyplot as plt


def lr_scheduler():
    def scheduler(epoch, lr):
        if (epoch + 1) % 100 == 0:
            return max(lr * 0.5, 1e-5)
        return lr

    return tf.keras.callbacks.LearningRateScheduler(scheduler)


def decompose(file_path, display_plot=True):
    assert os.path.isfile(file_path), "file not found"
    audio, sample_rate = sf.read(file_path)
    # audio, sample_rate = sf.read('samples/guitar_synthetic_000-060-025.wav')
    # audio, sample_rate = sf.read('samples/guitar_acoustic_019-060-075.wav')

    audio = tf.cast(audio, dtype=tf.float32)
    audio = tf.reshape(audio, shape=(1, -1))

    n_samples = audio.shape[1]
    frame_step = 64
    note_number = 60
    f0 = ddsp.core.midi_to_hz(note_number)
    n_harmonics = int(sample_rate / (2.0 * f0))
    harmonic_frequencies = ddsp.core.get_harmonic_frequencies(
        f0, n_harmonics)

    harmonic_model = None
    epochs = 500
    refinement_steps = 1

    for _ in range(refinement_steps):
        harmonic_model = sm.HarmonicModel(
            harmonic_frequencies, n_samples, sample_rate, frame_step)

        harmonic_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss=sm.ResidualError(),
            run_eagerly=False)

        harmonic_model.fit(audio, audio,
                           epochs=epochs,
                           callbacks=[lr_scheduler()])

        harmonic_frequencies = harmonic_model.harmonic_frequencies

    harmonic = harmonic_model(audio)
    residual = audio - harmonic

    output_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    file_name, ext = os.path.splitext(file_name)

    sf.write(os.path.join(output_dir, f'{file_name}_original{ext}'), np.squeeze(audio.numpy()), sample_rate)
    sf.write(os.path.join(output_dir, f'{file_name}_harmonic{ext}'), np.squeeze(harmonic.numpy()), sample_rate)
    sf.write(os.path.join(output_dir, f'{file_name}_residual{ext}'), np.squeeze(residual.numpy()), sample_rate)

    if display_plot:
        for i in range(3):  # range(harmonic_model.n_harmonics):
            phase_diff = harmonic_model.harmonic_phase_diffs[:, :, i]
            amplitude = harmonic_model.harmonic_amplitudes[:, :, i]
            phase_shift = harmonic_model.harmonic_phase_shifts[:, :, i]

            phase_diff = np.squeeze(phase_diff.numpy())
            amplitude = np.squeeze(amplitude.numpy())
            phase_shift = np.squeeze(phase_shift.numpy())

            plt.figure(i)
            plt.subplot(3, 1, 1)
            plt.plot(phase_diff)
            plt.subplot(3, 1, 2)
            plt.plot(amplitude)
            plt.subplot(3, 1, 3)
            plt.plot(phase_shift)

        plt.show()


if __name__ == '__main__':
    decompose('samples/guitar_electronic_003-060-050.wav')
