import tensorflow as tf
import numpy as np
import ddsp


class PhaseRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self,
                 dx1_amount=0.0,
                 dx1_threshold=0.0,
                 dx2_amount=0.1,
                 dx2_threshold=0.0):
        self.dx1_amount = dx1_amount
        self.dx1_threshold = dx1_threshold
        self.dx2_amount = dx2_amount
        self.dx2_threshold = dx2_threshold

    def __call__(self, x):
        dx1 = x[:, 1:, :] - x[:, :-1, :]
        dx1 = tf.where(dx1 > 0.5, dx1 - 1.0, dx1)
        dx1 = tf.where(dx1 < -0.5, dx1 + 1.0, dx1)

        loss = 0.0

        if self.dx1_amount > 0.0:
            y = tf.math.abs(dx1)
            y = tf.where(
                y < self.dx1_threshold,
                0.0,
                self.dx1_amount * (y - self.dx1_threshold))
            loss += tf.math.reduce_mean(y)

        if self.dx2_amount > 0.0:
            dx2 = dx1[:, 1:, :] - dx1[:, :-1, :]
            y = tf.math.abs(dx2)
            y = tf.where(
                y < self.dx2_threshold,
                0.0,
                self.dx2_amount * (y - self.dx2_threshold))
            loss += tf.math.reduce_mean(y)

        return loss

    def get_config(self):
        pass


class ResidualError(tf.keras.losses.Loss):
    def __init__(self, loss_type=1,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='residual_error'):
        super(ResidualError, self).__init__(
            reduction=reduction,
            name=name)
        self.loss_type = loss_type

    def call(self, y_true, y_pred):
        loss = 0.0
        if self.loss_type == 0:
            residual = (y_true - y_pred)
            loss += tf.reduce_mean(tf.math.square(residual), axis=-1)
        elif self.loss_type == 1:
            residual = (y_true - y_pred) / (tf.math.abs(y_true) + 1.0)
            loss += tf.reduce_mean(tf.math.square(residual), axis=-1)

        return loss


class HarmonicModel(tf.keras.Model):
    def __init__(self, harmonic_frequencies, n_samples, sample_rate, frame_step,
                 n_audio=1):
        super(HarmonicModel, self).__init__()
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.frame_step = frame_step
        self.n_frames = n_samples // frame_step
        self.n_harmonics = harmonic_frequencies.shape[-1]

        self._harmonic_frequencies = harmonic_frequencies

        self._harmonic_phase_shifts = self.add_weight(
            name='harmonic_phase_shifts',
            shape=(n_audio, self.n_frames + 1, self.n_harmonics),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            regularizer=PhaseRegularizer(dx1_amount=0.00,
                                         dx1_threshold=0.0,
                                         dx2_amount=0.001,
                                         dx2_threshold=0.0))

        self._harmonic_amplitudes = self.add_weight(
            name='harmonic_amplitudes',
            shape=(n_audio, self.n_frames, self.n_harmonics),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(1.0 / self.n_harmonics),
            trainable=True)

    @property
    def harmonic_phase_shifts(self):
        return self._harmonic_phase_shifts

    @property
    def harmonic_amplitudes(self):
        return tf.math.abs(self._harmonic_amplitudes)

    @property
    def harmonic_phase_diffs(self):
        hps = self.harmonic_phase_shifts
        diff = hps[:, 1:, :] - hps[:, :-1, :]
        diff = tf.where(diff > 0.5, diff - 1.0, diff)
        diff = tf.where(diff < -0.5, diff + 1.0, diff)

        return diff

    @property
    def harmonic_frequencies(self):
        frame_step = self.frame_step
        sample_rate = self.sample_rate

        diff = self.harmonic_phase_diffs
        hfs = diff * sample_rate / frame_step

        hf = self._harmonic_frequencies
        harmonic_frequencies = hf + hfs

        return harmonic_frequencies

    def call(self, audio, training=None, mask=None):
        n_samples = self.n_samples
        sample_rate = self.sample_rate

        initial_phase = self.harmonic_phase_shifts[:, :1, :]
        harmonic_frequencies = self.harmonic_frequencies
        harmonic_amplitudes = self.harmonic_amplitudes

        frequency_envelopes = ddsp.core.resample(
            harmonic_frequencies, n_samples)

        amplitude_envelopes = ddsp.core.resample(
            harmonic_amplitudes, n_samples)

        # Don't exceed Nyquist.
        amplitude_envelopes = ddsp.core.remove_above_nyquist(
            frequency_envelopes,
            amplitude_envelopes,
            sample_rate)

        # Angular frequency, Hz -> radians per sample.
        omegas = frequency_envelopes * (2.0 * np.pi) / float(sample_rate)
        phases = ddsp.core.angular_cumsum(omegas) + initial_phase

        # Convert to waveforms.
        wavs = tf.sin(phases)
        audio = amplitude_envelopes * wavs  # [mb, n_samples, n_sinusoids]
        audio = tf.reduce_sum(audio, axis=-1)  # [mb, n_samples]
        return audio

    def get_config(self):
        pass
