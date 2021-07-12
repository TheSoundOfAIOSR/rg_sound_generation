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


def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-6):
    return max_value * tf.nn.sigmoid(x)**tf.math.log(exponent) + threshold


class DataHandler:
    def __init__(self,
                 fix_pitch=True,
                 normalize_mag=False,
                 f0_weight='mag_max_pool',  # 'mag_max_pool', 'mag', 'none'
                 mag_env_weight='none',
                 h_freq_shifts_weight='mag_max_pool',
                 h_mag_weight='mag_max_pool',
                 mag_loss_mode='l2_db',  # ' l2_db' 'l1_db', 'rms_db', 'mse'
                 mag_scale_fn=exp_sigmoid,
                 min_f0=82.41,
                 sample_rate=16000,
                 max_semitone_displacement=1,
                 db_limit=-120.0,
                 max_pool_length=5):
        self._fix_pitch = fix_pitch
        self._normalize_mag = normalize_mag
        self._f0_weight = f0_weight
        self._mag_env_weight = mag_env_weight
        self._h_freq_shifts_weight = h_freq_shifts_weight
        self._h_mag_weight = h_mag_weight
        self._mag_loss_mode = mag_loss_mode
        self._mag_scale_fn = mag_scale_fn
        self._sample_rate = sample_rate
        self._f0_st_factor = 2.0 ** (max_semitone_displacement / 12.0) - 1.0
        self._db_limit = db_limit
        self._lin_limit = tsms.core.db_to_lin(db_limit)
        self._max_pool_length = max_pool_length

        # self.max_harmonics = tsms.core.get_number_harmonics(
        #     min_f0, sample_rate)
        self.max_harmonics = 110

    @tf.function
    def normalize(self, h_freq, h_mag, note_number, name=None):
        note_number = tf.cast(note_number, dtype=tf.float32)
        f0 = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)[:, :, tf.newaxis]
        f0_note = tsms.core.midi_to_hz(note_number)
        max_f0 = f0_note * self._f0_st_factor

        if self._normalize_mag:
            h_mag = h_mag / tf.math.reduce_max(h_mag)

        mag_env = tf.math.reduce_sum(h_mag, axis=2, keepdims=True)
        mag_env = tf.where(mag_env < self._lin_limit, self._lin_limit, mag_env)

        batches = tf.shape(h_freq)[0]
        frames = tf.shape(h_freq)[1]
        harmonics = tf.shape(h_freq)[2]
        harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
        harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

        h_freq_shifts = (h_freq / harmonic_numbers - f0) / max_f0
        h_mag_distribution = h_mag / mag_env

        h_freq_shifts = tf.pad(
            h_freq_shifts,
            paddings=((0, 0), (0, 0), (0, self.max_harmonics - harmonics)))

        h_mag_distribution = tf.pad(
            h_mag_distribution,
            paddings=((0, 0), (0, 0), (0, self.max_harmonics - harmonics)))

        if self._fix_pitch:
            f0_mean = tf.math.reduce_mean(f0)
            f0_shifts = (f0 - f0_mean) / max_f0
            harmonics = tsms.core.get_number_harmonics(
                f0_mean, self._sample_rate)
        else:
            f0_shifts = (f0 - f0_note) / max_f0

        mask = tf.concat([
            tf.ones(shape=(batches, frames, harmonics)),
            tf.zeros(shape=(batches, frames, self.max_harmonics - harmonics))],
            axis=2)

        h_freq_shifts *= mask
        h_mag_distribution *= mask

        return f0_shifts, mag_env, h_freq_shifts, h_mag_distribution, mask

    def denormalize(self, f0_shifts, mag_env,
                    h_freq_shifts, h_mag_distribution, mask,
                    note_number):
        note_number = tf.cast(note_number, dtype=tf.float32)
        f0_note = tsms.core.midi_to_hz(note_number)
        max_f0 = f0_note * self._f0_st_factor

        harmonics = tf.shape(h_freq_shifts)[2]
        harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
        harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

        h_freq_shifts *= mask
        h_mag_distribution *= mask

        f0 = f0_note + f0_shifts * max_f0
        h_freq = harmonic_numbers * (f0 + h_freq_shifts * max_f0)
        h_mag = h_mag_distribution * mag_env

        return h_freq, h_mag

    def loss(self,
             f0_shifts_true, f0_shifts_pred,
             mag_env_true, mag_env_pred,
             h_freq_shifts_true, h_freq_shifts_pred,
             h_mag_distribution_true, h_mag_distribution_pred,
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

        if self._mag_scale_fn is not None:
            mag_env_pred = exp_sigmoid(mag_env_pred)
            h_mag_distribution_pred = exp_sigmoid(h_mag_distribution_pred)

        mag0 = mag_env_true
        mag1 = mag_env_true * h_mag_distribution_true

        f_w0 = compute_mag_weight(mag0, 1.0, self._f0_weight)
        f_w1 = compute_mag_weight(mag1, mask, self._h_freq_shifts_weight)
        m_w0 = compute_mag_weight(mag0, 1.0, self._mag_env_weight)
        m_w1 = compute_mag_weight(mag1, mask, self._h_mag_weight)

        f0_loss = tf.square(f0_shifts_true - f0_shifts_pred) * f_w0
        f0_loss = tf.math.reduce_sum(f0_loss) / tf.math.reduce_sum(f_w0)

        h_freq_shifts_loss = tf.square(
            h_freq_shifts_true - h_freq_shifts_pred) * f_w1
        h_freq_shifts_loss = tf.math.reduce_sum(
            h_freq_shifts_loss) / tf.math.reduce_sum(f_w1)

        mag_env_loss = 0.0
        h_mag_loss = 0.0
        if self._mag_loss_mode == 'mse':
            mag_env_loss = tf.math.square(mag_env_true - mag_env_pred) * m_w0
            h_mag_loss = tf.math.square(
                h_mag_distribution_true - h_mag_distribution_pred) * m_w1

        elif self._mag_loss_mode == 'rms_db':
            mag_env_loss = linear_to_normalized_db(
                tf.math.abs(mag_env_true - mag_env_pred)) * m_w0
            h_mag_loss = linear_to_normalized_db(tf.math.abs(
                h_mag_distribution_true - h_mag_distribution_pred)) * m_w1

        elif self._mag_loss_mode == 'l1_db':
            mag_env_true = linear_to_normalized_db(mag_env_true, self._db_limit)
            mag_env_pred = linear_to_normalized_db(mag_env_pred, self._db_limit)
            h_mag_distribution_true = linear_to_normalized_db(
                h_mag_distribution_true, self._db_limit)
            h_mag_distribution_pred = linear_to_normalized_db(
                h_mag_distribution_pred, self._db_limit)

            mag_env_loss = tf.math.abs(mag_env_true - mag_env_pred) * m_w0
            h_mag_loss = tf.math.abs(
                h_mag_distribution_true - h_mag_distribution_pred) * m_w1

        elif self._mag_loss_mode == 'l2_db':
            mag_env_true = linear_to_normalized_db(mag_env_true, self._db_limit)
            mag_env_pred = linear_to_normalized_db(mag_env_pred, self._db_limit)
            h_mag_distribution_true = linear_to_normalized_db(
                h_mag_distribution_true, self._db_limit)
            h_mag_distribution_pred = linear_to_normalized_db(
                h_mag_distribution_pred, self._db_limit)

            mag_env_loss = tf.math.square(mag_env_true - mag_env_pred) * m_w0
            h_mag_loss = tf.math.square(
                h_mag_distribution_true - h_mag_distribution_pred) * m_w1

        mag_env_loss = tf.math.reduce_mean(
            mag_env_loss) / tf.math.reduce_sum(m_w0)
        h_mag_loss = tf.math.reduce_sum(
            h_mag_loss) / tf.math.reduce_sum(m_w1)

        return f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss
