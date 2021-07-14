import tensorflow as tf
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
    return max_value * tf.nn.sigmoid(x)**tf.math.log(exponent) + threshold


class DataHandler:
    def __init__(self,
                 fix_pitch=True,
                 normalize_mag=False,
                 f0_weight_type='mag_max_pool',  # 'mag_max_pool', 'mag', 'none'
                 h_freq_shifts_weight_type='mag_max_pool',
                 mag_loss_type='l2_db',  # ' l2_db' 'l1_db', 'rms_db', 'mse'
                 f0_weight=1.0,
                 mag_env_weight=1.0,
                 h_freq_shifts_weight=1.0,
                 h_mag_dist_weight=1.0,
                 mag_scale_fn=exp_sigmoid,
                 max_harmonics=110,
                 sample_rate=16000,
                 max_semitone_displacement=1,
                 db_limit=-120.0,
                 max_pool_length=5):
        self._fix_pitch = fix_pitch
        self._normalize_mag = normalize_mag
        self._f0_weight_type = f0_weight_type
        self._h_freq_shifts_weight_type = h_freq_shifts_weight_type
        self._mag_loss_type = mag_loss_type
        self._f0_weight = f0_weight
        self._mag_env_weight = mag_env_weight
        self._h_freq_shifts_weight = h_freq_shifts_weight
        self._h_mag_dist_weight = h_mag_dist_weight
        self._mag_scale_fn = mag_scale_fn
        self._sample_rate = sample_rate
        self._f0_st_factor = 2.0 ** (max_semitone_displacement / 12.0) - 1.0
        self._db_limit = db_limit
        self._lin_limit = tsms.core.db_to_lin(db_limit)
        self._max_pool_length = max_pool_length
        self.max_harmonics = max_harmonics

    @property
    def f0_weight_type(self):
        return self._f0_weight_type

    @f0_weight_type.setter
    def f0_weight_type(self, value: str):
        assert value in ['mag_max_pool', 'mag', 'none']
        self._f0_weight_type = value

    @property
    def mag_loss_type(self):
        return self._mag_loss_type

    @mag_loss_type.setter
    def mag_loss_type(self, value: str):
        assert value in ['l2_db' 'l1_db', 'rms_db', 'mse']
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

    @tf.function
    def normalize(self, h_freq, h_mag, note_number, name=None):
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

        return f0_shifts, mag_env, h_freq_shifts, h_mag_dist, mask

    def scale_pred(self, f0_shifts_pred, mag_env_pred,
                   h_freq_shifts_pred, h_mag_dist_pred):
        if self._mag_scale_fn is not None:
            mag_env_pred = exp_sigmoid(mag_env_pred)
            h_mag_dist_pred = exp_sigmoid(h_mag_dist_pred)

        return f0_shifts_pred, mag_env_pred, h_freq_shifts_pred, h_mag_dist_pred

    def denormalize(self, f0_shifts, mag_env,
                    h_freq_shifts, h_mag_dist, mask,
                    note_number, pred=True):
        if pred:
            f0_shifts, mag_env, h_freq_shifts, h_mag_dist = \
                self.scale_pred(f0_shifts, mag_env, h_freq_shifts, h_mag_dist)

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

        f0_estimate = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)
        min_f0 = tf.math.reduce_min(f0_estimate)
        harmonics = tsms.core.get_number_harmonics(min_f0, self._sample_rate)

        h_freq = h_freq[:, :, :harmonics]
        h_mag = h_mag[:, :, :harmonics]

        return h_freq, h_mag

    @tf.function
    def loss(self,
             f0_shifts_true, f0_shifts_pred,
             mag_env_true, mag_env_pred,
             h_freq_shifts_true, h_freq_shifts_pred,
             h_mag_dist_true, h_mag_dist_pred,
             mask):

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
        f0_shifts_pred, mag_env_pred, h_freq_shifts_pred, h_mag_dist_pred = \
            self.scale_pred(f0_shifts_pred, mag_env_pred,
                            h_freq_shifts_pred, h_mag_dist_pred)

        mag0 = mag_env_true
        mag1 = mag_env_true * h_mag_dist_true

        f_w0 = compute_mag_weight(mag0, 1.0, self._f0_weight_type)
        f_w1 = compute_mag_weight(mag1, mask, self._h_freq_shifts_weight_type)
        m_w0 = 1.0
        m_w1 = mask

        f0_loss = tf.square(f0_shifts_true - f0_shifts_pred) * f_w0
        f0_loss = tf.math.reduce_sum(f0_loss) / tf.math.reduce_sum(f_w0)

        h_freq_shifts_loss = tf.square(
            h_freq_shifts_true - h_freq_shifts_pred) * f_w1
        h_freq_shifts_loss = tf.math.reduce_sum(
            h_freq_shifts_loss) / tf.math.reduce_sum(f_w1)

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

        f0_loss *= self._f0_weight
        mag_env_loss *= self._mag_env_weight
        h_freq_shifts_loss *= self._h_freq_shifts_weight
        h_mag_loss *= self._h_mag_dist_weight

        return f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss
