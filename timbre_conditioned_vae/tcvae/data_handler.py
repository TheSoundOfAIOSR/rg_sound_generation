import tensorflow as tf
import numpy as np
import tsms


def linear_to_normalized_db(x, db_limit=-120.0):
    y = tsms.core.lin_to_db(x)
    y = (tf.maximum(y, db_limit) - db_limit) / (-db_limit)
    return y


def normalized_db_to_linear(x, db_limit=-120.0):
    y = x * (-db_limit) + db_limit
    y = tsms.core.db_to_lin(y)
    return y


def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=0.0):
    return max_value * tf.nn.sigmoid(x) ** tf.math.log(exponent) + threshold


class SimpleDataHandler:
    def __init__(self,
                 sample_rate=16000,
                 frame_step=64,
                 frames=1001,
                 max_harmonics=110):
        self._sample_rate = sample_rate
        self._frame_step = frame_step
        self._frames = frames
        self._max_harmonics = max_harmonics
        self._f0_st_factor = 2.0 ** (1.0 / 12.0) - 1.0

    def normalize(self, h_freq, h_mag, h_phase, note_number):
        note_number = tf.cast(note_number, dtype=tf.float32)
        f0_note = tsms.core.midi_to_hz(note_number)
        max_f0_displ = f0_note * self._f0_st_factor

        batches = tf.shape(h_freq)[0]
        frames = tf.shape(h_freq)[1]
        harmonics = tf.shape(h_freq)[2]
        harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
        harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

        h_freq_shifts = (h_freq / harmonic_numbers - f0_note) / max_f0_displ

        normalized_data = {
            "h_freq_shifts": h_freq_shifts,
            "h_mag": h_mag,
        }

        mask = tf.concat([
            tf.ones((batches, frames, harmonics)),
            tf.zeros((batches, frames, self._max_harmonics - harmonics))],
            axis=2)

        return normalized_data, mask

    def denormalize(self, normalized_data, mask, note_number):
        h_freq_shifts = normalized_data["h_freq_shifts"]
        h_mag = normalized_data["h_mag"]

        note_number = tf.cast(note_number, dtype=tf.float32)
        f0_note = tsms.core.midi_to_hz(note_number)
        max_f0_displ = f0_note * self._f0_st_factor

        harmonics = tf.cast(tf.math.reduce_sum(mask[0, 0, :]), dtype=tf.int64)
        # harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
        harmonic_numbers = tf.range(1, self._max_harmonics + 1, dtype=tf.float32)
        harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]
        h_freq = (h_freq_shifts * max_f0_displ + f0_note) * harmonic_numbers

        h_freq = h_freq[:, :, :harmonics]
        h_mag = h_mag[:, :, :harmonics]

        h_phase = tsms.core.generate_phase(
            h_freq,
            sample_rate=self._sample_rate,
            frame_step=self._frame_step)

        return h_freq, h_mag, h_phase

    def input_transform(self, normalized_data, rows=1001, cols=110):
        h_freq_shifts = normalized_data["h_freq_shifts"]
        h_mag = normalized_data["h_mag"]

        frames = self._frames
        harmonics = tf.shape(h_freq_shifts)[2]

        h_freq_shifts = tf.pad(
            h_freq_shifts, ((0, 0), (0, rows - frames), (0, cols - harmonics)))
        h_mag = tf.pad(
            h_mag, ((0, 0), (0, rows - frames), (0, cols - harmonics)))

        h_freq_shifts = tf.expand_dims(h_freq_shifts, axis=-1)
        h_mag = tf.expand_dims(h_mag, axis=-1)
        h = tf.concat([h_freq_shifts, h_mag], axis=-1)
        return h

    def output_transform(self, h, pred=True):
        frames = self._frames
        max_harmonics = self._max_harmonics

        h_freq_shifts, h_mag = tf.unstack(h, axis=-1)

        h_freq_shifts = h_freq_shifts[:, :frames, :max_harmonics]
        h_mag = h_mag[:, :frames, :max_harmonics]

        normalized_data = {
            "h_freq_shifts": h_freq_shifts,
            "h_mag": h_mag,
        }

        return normalized_data

    def loss(self, normalized_data_true, normalized_data_pred, mask):
        h_freq_shifts_true = normalized_data_true["h_freq_shifts"]
        h_mag_true = normalized_data_true["h_mag"]

        h_freq_shifts_pred = normalized_data_pred["h_freq_shifts"]
        h_mag_true_pred = normalized_data_pred["h_mag"]

        # compute frequencies loss
        h_freq_shifts_loss = tf.math.square(
            h_freq_shifts_true - h_freq_shifts_pred) * h_mag_true
        h_freq_shifts_loss = tf.math.reduce_sum(
            h_freq_shifts_loss) / tf.math.reduce_sum(h_mag_true)

        # compute magnitude loss
        h_mag_loss = tf.math.square(
            h_mag_true - h_mag_true_pred) * mask
        h_mag_loss = tf.math.reduce_sum(
            h_mag_loss) / tf.math.reduce_sum(mask)

        loss = h_freq_shifts_loss + h_mag_loss

        losses = {
            "loss": loss,
            "h_freq_shifts_loss": h_freq_shifts_loss,
            "h_mag_loss": h_mag_loss,
        }

        return losses


class DataHandler:
    def __init__(self,
                 fix_pitch=True,
                 normalize_mag=False,
                 use_phase=False,
                 weight_type='mag_max_pool',  # 'mag_max_pool', 'mag', 'none'
                 mag_loss_type='l2_db',  # ' l2_db' 'l1_db', 'rms_db', 'mse'
                 f0_weight=1.0,
                 mag_env_weight=1.0,
                 h_freq_shifts_weight=1.0,
                 h_mag_dist_weight=1.0,
                 h_phase_diff_weight=1.0,
                 mag_scale_fn=exp_sigmoid,
                 max_harmonics=110,
                 sample_rate=16000,
                 frame_step=64,
                 frames=1001,
                 max_semitone_displacement=1,
                 db_limit=-120.0,
                 max_pool_length=5):
        self._fix_pitch = fix_pitch
        self._normalize_mag = normalize_mag
        self._use_phase = use_phase
        self._weight_type = weight_type
        self._mag_loss_type = mag_loss_type
        self._f0_weight = f0_weight
        self._mag_env_weight = mag_env_weight
        self._h_freq_shifts_weight = h_freq_shifts_weight
        self._h_mag_dist_weight = h_mag_dist_weight
        self._h_phase_diff_weight = h_phase_diff_weight
        self._mag_scale_fn = mag_scale_fn
        self._sample_rate = sample_rate
        self._frame_step = frame_step
        self._frames = frames
        self._f0_st_factor = 2.0 ** (max_semitone_displacement / 12.0) - 1.0
        self._db_limit = db_limit
        self._lin_limit = tsms.core.db_to_lin(db_limit)
        self._max_pool_length = max_pool_length
        self.max_harmonics = max_harmonics

    @property
    def use_phase(self):
        return self._use_phase

    @use_phase.setter
    def use_phase(self, value: bool):
        self._use_phase = value

    @property
    def weight_type(self):
        return self._weight_type

    @weight_type.setter
    def weight_type(self, value: str):
        assert value in ['mag_max_pool', 'mag', 'none']
        self._weight_type = value

    @property
    def mag_loss_type(self):
        return self._mag_loss_type

    @mag_loss_type.setter
    def mag_loss_type(self, value: str):
        assert value in ['l2_db', 'l1_db', 'rms_db', 'mse']
        self._mag_loss_type = value

    @property
    def f0_weight(self):
        return self._f0_weight

    @f0_weight.setter
    def f0_weight(self, value: float):
        self._f0_weight = value

    @property
    def mag_env_weight(self):
        return self._mag_env_weight

    @mag_env_weight.setter
    def mag_env_weight(self, value: float):
        self._mag_env_weight = value

    @property
    def h_freq_shifts_weight(self):
        return self._h_freq_shifts_weight

    @h_freq_shifts_weight.setter
    def h_freq_shifts_weight(self, value: float):
        self._h_freq_shifts_weight = value

    @property
    def h_mag_dist_weight(self):
        return self._h_mag_dist_weight

    @h_mag_dist_weight.setter
    def h_mag_dist_weight(self, value: float):
        self._h_mag_dist_weight = value

    @property
    def h_phase_diff_weight(self):
        return self._h_phase_diff_weight

    @h_phase_diff_weight.setter
    def h_phase_diff_weight(self, value: float):
        self._h_phase_diff_weight = value

    @property
    def mag_scale_fn(self):
        if self._mag_scale_fn is None:
            return "none"
        else:
            return "exp_sigmoid"

    @mag_scale_fn.setter
    def mag_scale_fn(self, value: str):
        assert value in ["none", "exp_sigmoid"]
        if value == "none":
            self._mag_scale_fn = None
        else:
            self._mag_scale_fn = exp_sigmoid

    def normalize(self, h_freq, h_mag, h_phase, note_number):
        note_number = tf.cast(note_number, dtype=tf.float32)
        f0 = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)[:, :, tf.newaxis]
        f0_note = tsms.core.midi_to_hz(note_number)
        max_f0_displ = f0_note * self._f0_st_factor

        if self._normalize_mag:
            h_mag = h_mag / tf.math.reduce_max(h_mag)

        mag_env = tf.math.reduce_sum(h_mag, axis=2, keepdims=True)
        mag_env = tf.where(mag_env < self._lin_limit, self._lin_limit, mag_env)

        batches = tf.shape(h_freq)[0]
        frames = tf.shape(h_freq)[1]
        harmonics = tf.shape(h_freq)[2]
        harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
        harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

        h_freq_shifts = (h_freq / harmonic_numbers - f0) / max_f0_displ
        h_mag_dist = h_mag / mag_env

        h_freq_shifts = tf.pad(
            h_freq_shifts,
            paddings=((0, 0), (0, 0), (0, self.max_harmonics - harmonics)))

        h_mag_dist = tf.pad(
            h_mag_dist,
            paddings=((0, 0), (0, 0), (0, self.max_harmonics - harmonics)))

        if self._use_phase:
            h_phase_gen = tsms.core.generate_phase(
                h_freq,
                sample_rate=self._sample_rate,
                frame_step=self._frame_step)
            h_phase_diff = tsms.core.phase_diff(h_phase, h_phase_gen)
            # unwrap d_phase from +/- pi to +/- 2*pi
            h_phase_diff = tsms.core.phase_unwrap(h_phase_diff, axis=1)
            h_phase_diff = (h_phase_diff + 2.0 * np.pi) % \
                           (4.0 * np.pi) - 2.0 * np.pi
            h_phase_diff /= (4.0 * np.pi)
        else:
            h_phase_diff = tf.zeros_like(h_freq)

        h_phase_diff = tf.pad(
            h_phase_diff,
            paddings=((0, 0), (0, 0), (0, self.max_harmonics - harmonics)))

        if self._fix_pitch:
            f0_mean = tf.math.reduce_mean(f0)
            f0_shifts = (f0 - f0_mean) / max_f0_displ
            harmonics = tsms.core.get_number_harmonics(
                f0_mean, self._sample_rate)
        else:
            f0_shifts = (f0 - f0_note) / max_f0_displ

        mask = tf.concat([
            tf.ones(shape=(batches, frames, harmonics)),
            tf.zeros(shape=(batches, frames, self.max_harmonics - harmonics))],
            axis=2)

        h_freq_shifts *= mask
        h_mag_dist *= mask
        h_phase_diff *= mask

        normalized_data = {
            "f0_shifts": f0_shifts,
            "mag_env": mag_env,
            "h_freq_shifts": h_freq_shifts,
            "h_mag_dist": h_mag_dist,
            "h_phase_diff": h_phase_diff,
        }

        return normalized_data, mask

    def denormalize(self, normalized_data, mask, note_number):
        f0_shifts = normalized_data["f0_shifts"]
        mag_env = normalized_data["mag_env"]
        h_freq_shifts = normalized_data["h_freq_shifts"]
        h_mag_dist = normalized_data["h_mag_dist"]
        h_phase_diff = normalized_data["h_phase_diff"]

        note_number = tf.cast(note_number, dtype=tf.float32)
        f0_note = tsms.core.midi_to_hz(note_number)
        max_f0_displ = f0_note * self._f0_st_factor

        harmonics = tf.shape(h_freq_shifts)[2]
        harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
        harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

        h_freq_shifts *= mask
        h_mag_dist *= mask

        f0 = f0_note + f0_shifts * max_f0_displ
        h_freq = harmonic_numbers * (f0 + h_freq_shifts * max_f0_displ)
        h_mag = h_mag_dist * mag_env

        h_phase = tsms.core.generate_phase(
            h_freq,
            sample_rate=self._sample_rate,
            frame_step=self._frame_step)

        if self._use_phase:
            h_phase = (h_phase + h_phase_diff) % (2.0 * np.pi)

        f0_estimate = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)
        min_f0 = tf.math.reduce_min(f0_estimate)
        harmonics = tsms.core.get_number_harmonics(min_f0, self._sample_rate)

        h_freq = h_freq[:, :, :harmonics]
        h_mag = h_mag[:, :, :harmonics]
        h_phase = h_phase[:, :, :harmonics]

        return h_freq, h_mag, h_phase

    def input_transform(self, normalized_data, rows=1001, cols=110):
        f0_shifts = normalized_data["f0_shifts"]
        mag_env = normalized_data["mag_env"]
        h_freq_shifts = normalized_data["h_freq_shifts"]
        h_mag_dist = normalized_data["h_mag_dist"]
        h_phase_diff = normalized_data["h_phase_diff"]

        frames = self._frames
        harmonics = tf.shape(h_freq_shifts)[2]

        freq = tf.concat([f0_shifts, h_freq_shifts], axis=-1)
        mag = tf.concat([mag_env, h_mag_dist], axis=-1)

        freq = tf.pad(
            freq, ((0, 0), (0, rows - frames), (0, cols - harmonics - 1)))
        mag = tf.pad(
            mag, ((0, 0), (0, rows - frames), (0, cols - harmonics - 1)))

        freq = tf.expand_dims(freq, axis=-1)
        mag = tf.expand_dims(mag, axis=-1)

        h = tf.concat([freq, mag], axis=-1)

        if self._use_phase:
            h_phase_diff = tf.pad(
                h_phase_diff,
                ((0, 0), (0, rows - frames), (0, cols - harmonics)))
            h_phase_diff = tf.expand_dims(h_phase_diff, axis=-1)

            h = tf.concat([h, h_phase_diff], axis=-1)

        return h

    def output_transform(self, h, pred=True):
        batches = tf.shape(h)[0]
        frames = self._frames
        max_harmonics = self.max_harmonics

        if self._use_phase:
            freq, mag, h_phase_diff = tf.unstack(h, axis=-1)
            h_phase_diff = h_phase_diff[:, :frames, :max_harmonics]
        else:
            freq, mag = tf.unstack(h, axis=-1)
            h_phase_diff = tf.zeros(shape=(batches, frames, max_harmonics))

        f0_shifts = freq[:, :frames, :1]
        mag_env = mag[:, :frames, :1]
        h_freq_shifts = freq[:, :frames, 1:max_harmonics + 1]
        h_mag_dist = mag[:, :frames, 1:max_harmonics + 1]

        # scale outputs
        if pred:
            if self._mag_scale_fn is not None:
                mag_env = exp_sigmoid(mag_env)
                h_mag_dist = exp_sigmoid(h_mag_dist)

            if self._use_phase:
                h_phase_diff = h_phase_diff % 1.0

        normalized_data = {
            "f0_shifts": f0_shifts,
            "mag_env": mag_env,
            "h_freq_shifts": h_freq_shifts,
            "h_mag_dist": h_mag_dist,
            "h_phase_diff": h_phase_diff,
        }

        return normalized_data

    def loss(self, normalized_data_true, normalized_data_pred, mask):
        f0_shifts_true = normalized_data_true["f0_shifts"]
        mag_env_true = normalized_data_true["mag_env"]
        h_freq_shifts_true = normalized_data_true["h_freq_shifts"]
        h_mag_dist_true = normalized_data_true["h_mag_dist"]
        h_phase_diff_true = normalized_data_true["h_phase_diff"]

        f0_shifts_pred = normalized_data_pred["f0_shifts"]
        mag_env_pred = normalized_data_pred["mag_env"]
        h_freq_shifts_pred = normalized_data_pred["h_freq_shifts"]
        h_mag_dist_pred = normalized_data_pred["h_mag_dist"]
        h_phase_diff_pred = normalized_data_pred["h_phase_diff"]

        def compute_mag_weight(mag, mask, mode):
            weight = mask
            if mode == 'mag':
                weight *= mag
            elif mode == 'mag_max_pool':
                weight *= tf.nn.max_pool1d(
                    mag,
                    [1, self._max_pool_length, 1], 1,
                    padding='SAME', data_format='NWC')
            return weight

        # compute weights
        mag0 = mag_env_true
        mag1 = mag_env_true * h_mag_dist_true

        f_w0 = compute_mag_weight(mag0, 1.0, self._weight_type)
        f_w1 = compute_mag_weight(mag1, mask, self._weight_type)
        m_w0 = 1.0
        m_w1 = mask
        p_w = f_w1

        # compute frequencies loss
        f0_loss = tf.square(f0_shifts_true - f0_shifts_pred) * f_w0
        f0_loss = tf.math.reduce_sum(f0_loss) / tf.math.reduce_sum(f_w0)

        h_freq_shifts_loss = tf.square(
            h_freq_shifts_true - h_freq_shifts_pred) * f_w1
        h_freq_shifts_loss = tf.math.reduce_sum(
            h_freq_shifts_loss) / tf.math.reduce_sum(f_w1)

        # compute magnitudes loss
        mag_env_loss = 0.0
        h_mag_loss = 0.0

        h_mag_true = mag_env_true * h_mag_dist_true
        h_mag_pred = mag_env_pred * h_mag_dist_pred

        if self._mag_loss_type == 'mse':
            mag_env_loss = tf.math.square(mag_env_true - mag_env_pred) * m_w0
            h_mag_loss = tf.math.square(
                h_mag_true - h_mag_pred) * m_w1

        elif self._mag_loss_type == 'rms_db':
            mag_env_loss = linear_to_normalized_db(
                tf.math.abs(mag_env_true - mag_env_pred)) * m_w0
            h_mag_loss = linear_to_normalized_db(tf.math.abs(
                h_mag_true - h_mag_pred)) * m_w1

        elif self._mag_loss_type == 'l1_db':
            mag_env_true = linear_to_normalized_db(mag_env_true, self._db_limit)
            mag_env_pred = linear_to_normalized_db(mag_env_pred, self._db_limit)
            h_mag_true = linear_to_normalized_db(
                h_mag_true, self._db_limit)
            h_mag_pred = linear_to_normalized_db(
                h_mag_pred, self._db_limit)

            mag_env_loss = tf.math.abs(mag_env_true - mag_env_pred) * m_w0
            h_mag_loss = tf.math.abs(
                h_mag_true - h_mag_pred) * m_w1

        elif self._mag_loss_type == 'l2_db':
            mag_env_true = linear_to_normalized_db(mag_env_true, self._db_limit)
            mag_env_pred = linear_to_normalized_db(mag_env_pred, self._db_limit)
            h_mag_true = linear_to_normalized_db(
                h_mag_true, self._db_limit)
            h_mag_pred = linear_to_normalized_db(
                h_mag_pred, self._db_limit)

            mag_env_loss = tf.math.square(mag_env_true - mag_env_pred) * m_w0
            h_mag_loss = tf.math.square(
                h_mag_true - h_mag_pred) * m_w1

        mag_env_loss = tf.math.reduce_mean(
            mag_env_loss) / tf.math.reduce_sum(m_w0)
        h_mag_loss = tf.math.reduce_sum(
            h_mag_loss) / tf.math.reduce_sum(m_w1)

        # compute phase loss
        h_phase_diff_loss = 0.0
        if self._use_phase:
            h_phase_diff_loss = tf.square(
                h_phase_diff_true - h_phase_diff_pred) * p_w
            h_phase_diff_loss = tf.math.reduce_sum(
                h_phase_diff_loss) / tf.math.reduce_sum(p_w)

        # weight losses
        f0_loss *= self._f0_weight
        mag_env_loss *= self._mag_env_weight
        h_freq_shifts_loss *= self._h_freq_shifts_weight
        h_mag_loss *= self._h_mag_dist_weight
        h_phase_diff_loss *= self._h_phase_diff_weight

        loss = \
            f0_loss + \
            mag_env_loss + \
            h_freq_shifts_loss + \
            h_mag_loss + \
            h_phase_diff_loss

        losses = {
            "loss": loss,
            "f0_loss": f0_loss,
            "mag_env_loss": mag_env_loss,
            "h_freq_shifts_loss": h_freq_shifts_loss,
            "h_mag_loss": h_mag_loss,
            "h_phase_diff_loss": h_phase_diff_loss,
        }

        return losses
