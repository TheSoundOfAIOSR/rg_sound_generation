import tensorflow as tf
import numpy as np
from . import core


class ResidualError(tf.keras.losses.Loss):
    def __init__(self,
                 loss_type=0,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name='residual_error'):
        super(ResidualError, self).__init__(
            reduction=reduction,
            name=name)
        self.loss_type = loss_type

    def call(self, y_true, y_pred):
        loss = 0.0

        if self.loss_type == 0:
            y_pred = y_pred[:, :y_true.shape[1]]
            residual = y_true - y_pred
            loss += tf.reduce_mean(tf.math.square(residual), axis=-1)
        elif self.loss_type == 1:
            y_pred = y_pred[:, :y_true.shape[1]]
            residual = (y_true - y_pred) / (tf.math.abs(y_true) + 1.0)
            loss += tf.reduce_mean(tf.math.square(residual), axis=-1)

        return loss


class HarmonicModelRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, h_freq, h_mag, h_phase,
                 sample_rate, frame_step,
                 freq_amount=0.0, mag_amount=0.0, phase_amount=0.0):
        self._h_freq = h_freq
        self._h_mag = h_mag
        self._h_phase = h_phase
        self.sample_rate = sample_rate
        self.frame_step = frame_step
        self.freq_amount = freq_amount
        self.mag_amount = mag_amount
        self.phase_amount = phase_amount

    def __call__(self, x):
        loss = 0.0

        h_freq_shift = x[:, :, :, 0]
        h_mag_shift = x[:, :, :, 1]
        h_phase_shift = x[:, :, :, 2]

        frame_rate = self.sample_rate / self.frame_step
        h_freq = self._h_freq + frame_rate * h_freq_shift
        h_mag = self._h_mag + h_mag_shift
        h_phase = self._h_phase + 2.0 * np.pi * h_phase_shift

        if self.freq_amount > 0.0:
            freq_diff = h_freq[:, 1:, :] - h_freq[:, :-1, :]
            loss += tf.math.abs(freq_diff)

        if self.mag_amount > 0.0:
            mag_diff = h_mag[:, 1:, :] - h_mag[:, :-1, :]
            loss += tf.math.abs(mag_diff)

        if self.phase_amount > 0.0:
            g_phase = core.generate_phase(
                h_freq, self.sample_rate, self.frame_step)

            phase_diff = core.phase_diff(h_phase, g_phase)

            phase_diff = core.phase_diff(
                phase_diff[:, 1:, :],
                phase_diff[:, :-1, :])

            loss += tf.math.abs(phase_diff) / 2.0 * np.pi

        return loss

    def get_config(self):
        pass


class HarmonicModel(tf.keras.Model):
    def __init__(self, sample_rate, frame_step, channels, frames, harmonics,
                 h_freq=None, h_mag=None, h_phase=None, generate_phase=False):
        super(HarmonicModel, self).__init__()
        self.sample_rate = sample_rate
        self.frame_step = frame_step
        self.channels = channels
        self.frames = frames
        self.harmonics = harmonics

        if h_freq is None:
            h_freq = tf.zeros(shape=(1, 1, 1))
        if h_mag is None:
            h_mag = tf.zeros(shape=(1, 1, 1))
        if h_phase is None:
            h_phase = tf.zeros(shape=(1, 1, 1))

        self._h_freq = h_freq
        self._h_mag = h_mag
        self._h_phase = h_phase
        self.generate_phase = generate_phase

        self._shifts = self.add_weight(
            name='shifts',
            shape=(self.channels, self.frames, self.harmonics, 3),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Zeros(),
            trainable=True)
            # regularizer=HarmonicModelRegularizer(
            #     h_freq=self._h_freq, h_mag=self._h_mag, h_phase=self._h_phase,
            #     sample_rate=self.sample_rate, frame_step=self.frame_step,
            #     freq_amount=0.0, mag_amount=0.0, phase_amount=0.0))

    @property
    def h_freq_shift(self):
        return self._shifts[:, :, :, 0]

    @property
    def h_mag_shift(self):
        return self._shifts[:, :, :, 1]

    @property
    def h_phase_shift(self):
        return self._shifts[:, :, :, 2]

    @property
    def h_freq(self):
        frame_rate = self.sample_rate / self.frame_step
        return self._h_freq + frame_rate * self.h_freq_shift
    
    @h_freq.setter
    def h_freq(self, value):
        frame_rate = self.sample_rate / self.frame_step
        self._h_freq = value - frame_rate * self.h_freq_shift

    @property
    def h_mag(self):
        return self._h_mag + self.h_mag_shift

    @h_mag.setter
    def h_mag(self, value):
        self._h_mag = value - self.h_mag_shift

    @property
    def h_phase(self):
        return self._h_phase + 2.0 * np.pi * self.h_phase_shift

    @h_phase.setter
    def h_phase(self, value):
        self._h_phase = value - 2.0 * np.pi * self.h_phase_shift

    def call(self, inputs=None, training=None, mask=None):
        sample_rate = self.sample_rate
        frame_step = self.frame_step

        h_freq = self.h_freq
        h_phase = self.h_phase
        h_mag = self.h_mag

        if self.generate_phase:
            h_phase = core.generate_phase(h_freq, sample_rate, frame_step,
                                          initial_h_phase=None)

        audio = core.harmonic_synthesis(
            h_freq, h_mag, h_phase, sample_rate, frame_step)

        return audio

    def get_config(self):
        pass
