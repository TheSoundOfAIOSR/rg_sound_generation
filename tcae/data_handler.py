import tensorflow as tf
import numpy as np
import tsms
from tcae import heuristics


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


def mu_law_encode(signal, quantization_channels, range_0_1=False):
    # mu-law companding and mu-bits quantization
    mu = quantization_channels - 1
    mu = tf.cast(mu, dtype=tf.float32)

    if range_0_1:
        signal = tf.clip_by_value(signal, 0.0, 1.0)
    else:
        signal = tf.clip_by_value(signal, -1.0, 1.0)

    magnitude = tf.math.log1p(mu * tf.math.abs(signal)) / tf.math.log1p(mu)
    signal = tf.math.sign(signal) * magnitude

    if not range_0_1:
        # Map signal from [-1, +1] to [0, mu-1]
        signal = (signal + 1.0) / 2.0

    signal = signal * mu
    signal = tf.math.round(signal)
    quantized_signal = tf.cast(signal, dtype=tf.int32)

    return quantized_signal


def mu_law_decode(signal, quantization_channels, range_0_1=False):
    # inverse mu-law companding and dequantization
    mu = quantization_channels - 1
    mu = tf.cast(mu, dtype=tf.float32)
    y = tf.cast(signal, dtype=tf.float32)

    y = y / mu

    if not range_0_1:
        y = 2.0 * y - 1.0

    x = tf.math.sign(y) * (1.0 / mu) * ((1.0 + mu) ** tf.math.abs(y) - 1.0)

    return x


class DataHandler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DataHandler, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self,
                 fix_pitch=True,
                 normalize_mag=False,
                 compact_measures_logs=True,
                 weight_type='mag_max_pool',  # 'mag_max_pool', 'mag', 'none'
                 freq_loss_type='mse',  # 'cross_entropy', 'mse'
                 mag_loss_type='l2_db',  # 'cross_entropy', 'l2_db' 'l1_db', 'rms_db', 'mse'
                 freq_scale_fn=tf.math.tanh,
                 mag_scale_fn=exp_sigmoid,
                 phase_scale_fn=tf.math.tanh,
                 max_harmonics=110,
                 sample_rate=16000,
                 frame_step=64,
                 frames=1001,
                 max_semitone_displacement=2,
                 db_limit=-120.0,
                 h_mag_delta=1e-6,
                 max_pool_length=5,
                 quantization_channels=256,
                 freq_bands=None):
        self.fix_pitch = fix_pitch
        self.normalize_mag = normalize_mag
        self.compact_measures_logs = compact_measures_logs
        self._weight_type = weight_type
        self._freq_loss_type = freq_loss_type
        self._mag_loss_type = mag_loss_type
        self._mag_scale_fn = mag_scale_fn
        self.max_harmonics = max_harmonics
        self._sample_rate = sample_rate
        self._frame_step = frame_step
        self._frames = frames
        self._f0_st_factor = 2.0 ** (max_semitone_displacement / 12.0) - 1.0
        self._db_limit = db_limit
        self._lin_limit = tsms.core.db_to_lin(db_limit)
        self._h_mag_delta = h_mag_delta
        self._max_pool_length = max_pool_length
        self.quantization_channels = quantization_channels

        if freq_bands is None:
            self.freq_bands = {
                "bass": [60, 270],
                "mid": [270, 2000],
                "high_mid": [2000, 6000],
                "high": [6000, 20000]
            }
        else:
            self.freq_bands = freq_bands

        # obtained computing the max of the whole datasets
        self._mag_env_max = 6.4551239013671875

        self._losses_weights = None
        self._outputs = None
        self.measures_losses_weights = {
            "inharmonic": 0.1,
            "even_odd": 0.1,
            "sparse_rich": 0.1,
            "attack_rms": 0.1,
            "decay_rms": 0.1,
            "attack_time": 0.1,
            "decay_time": 0.1,
            "bass": 0.1,
            "mid": 0.1,
            "high_mid": 0.1,
            "high": 0.1,
        }

        self.update_losses_weights()

    @property
    def losses_weights(self):
        return self._losses_weights

    @property
    def outputs(self):
        return self._outputs

    def update_losses_weights(
            self,
            f0_shifts=1.0,
            h_freq_shifts=1.0,
            mag_env=1.0,
            h_mag_dist=1.0,
            h_mag=1.0,
            h_phase_diff=0.0,
            measures=0.0):

        self._losses_weights = {
            "f0_shifts": f0_shifts,
            "h_freq_shifts": h_freq_shifts,
            "mag_env": mag_env,
            "h_mag_dist": h_mag_dist,
            "h_mag": h_mag,
            "h_phase_diff": h_phase_diff,
            "measures": measures
        }
        self._outputs = {}
        for k, v in self._losses_weights.items():
            if v > 0.0:
                if k == "f0_shifts" or k == "mag_env":
                    self._outputs[k] = {"size": 1}
                elif k == "h_freq_shifts" or k == "h_mag_dist" or k == "h_phase_diff":
                    self._outputs[k] = {"size": self.max_harmonics}
                elif k == "h_mag":
                    self._outputs["mag_env"] = {"size": 1}
                    self._outputs["h_mag_dist"] = {"size": self.max_harmonics}
                elif k == "measures":
                    self._outputs["f0_shifts"] = {"size": 1}
                    self._outputs["h_freq_shifts"] = {"size": self.max_harmonics}
                    self._outputs["mag_env"] = {"size": 1}
                    self._outputs["h_mag_dist"] = {"size": self.max_harmonics}

    @property
    def weight_type(self):
        return self._weight_type

    @weight_type.setter
    def weight_type(self, value: str):
        assert value in ['mag_max_pool', 'mag', 'none']
        self._weight_type = value

    @property
    def freq_loss_type(self):
        return self._freq_loss_type

    @freq_loss_type.setter
    def freq_loss_type(self, value: str):
        assert value in ['cross_entropy', 'mse']
        self._freq_loss_type = value

    @property
    def mag_loss_type(self):
        return self._mag_loss_type

    @mag_loss_type.setter
    def mag_loss_type(self, value: str):
        assert value in ['cross_entropy', 'l2_db', 'l1_db', 'rms_db', 'mse']
        self._mag_loss_type = value

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

    def split_mag(self, h_mag):
        mag_env_max = self._mag_env_max
        mag_env = tf.math.reduce_sum(h_mag, axis=2, keepdims=True)
        h_mag_dist = h_mag / (mag_env + self._h_mag_delta)
        mag_env = mag_env / mag_env_max
        return mag_env, h_mag_dist

    def combine_mag(self, mag_env, h_mag_dist):
        mag_env_max = self._mag_env_max
        mag_env = mag_env * mag_env_max
        h_mag = h_mag_dist * (mag_env + self._h_mag_delta)
        return h_mag

    def normalize(self, h_freq, h_mag, h_phase, note_number):
        f0 = tsms.core.harmonic_analysis_to_f0(h_freq, h_mag)[:, :, tf.newaxis]
        note_number = tf.reshape(note_number, shape=(-1, 1, 1))
        f0_note = tsms.core.midi_to_hz(tf.cast(note_number, dtype=tf.float32))
        max_f0_displ = f0_note * self._f0_st_factor

        batches = tf.shape(h_freq)[0]
        frames = tf.shape(h_freq)[1]
        harmonics = tf.shape(h_freq)[2]
        harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
        harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

        # compute f0_shifts and h_freq_shifts
        if self.fix_pitch:
            f0_mean = tf.math.reduce_mean(f0)
            f0_shifts = (f0 - f0_mean) / max_f0_displ
            fixed_harmonics = tsms.core.get_number_harmonics(
                f0_mean, self._sample_rate)
        else:
            f0_shifts = (f0 - f0_note) / max_f0_displ
            fixed_harmonics = harmonics

        h_freq_shifts = ((h_freq / harmonic_numbers) - f0) / max_f0_displ

        f0_shifts = tf.clip_by_value(f0_shifts, -1.0, 1.0)
        h_freq_shifts = tf.clip_by_value(h_freq_shifts, -1.0, 1.0)

        # compute mag_env and h_mag_dist
        if self.normalize_mag:
            h_mag = h_mag / tf.math.reduce_max(h_mag)

        mag_env, h_mag_dist = self.split_mag(h_mag)

        # compute h_phase_diff
        h_phase_gen = tsms.core.generate_phase(
            h_freq,
            sample_rate=self._sample_rate,
            frame_step=self._frame_step)

        h_phase_diff = tsms.core.phase_diff(h_phase, h_phase_gen)
        # unwrap d_phase from +/- pi to +/- 2*pi
        h_phase_diff = tsms.core.phase_unwrap(h_phase_diff, axis=1)
        h_phase_diff = tf.where(
            tf.math.abs(h_phase_diff) > 2.0 * np.pi, 0.0, h_phase_diff)
        h_phase_diff = h_phase_diff / (2.0 * np.pi)  # [-1, +1] range

        # zero-padding to max_harmonics
        h_freq_shifts = tf.pad(
            h_freq_shifts,
            paddings=((0, 0), (0, 0), (0, self.max_harmonics - harmonics)))

        h_mag_dist = tf.pad(
            h_mag_dist,
            paddings=((0, 0), (0, 0), (0, self.max_harmonics - harmonics)))

        h_phase_diff = tf.pad(
            h_phase_diff,
            paddings=((0, 0), (0, 0), (0, self.max_harmonics - harmonics)))

        # compute mask
        mask = tf.concat([
            tf.ones((batches, frames, fixed_harmonics)),
            tf.zeros((batches, frames, self.max_harmonics - fixed_harmonics))],
            axis=2)

        h_freq_shifts = h_freq_shifts * mask
        h_mag_dist = h_mag_dist * mask
        h_phase_diff = h_phase_diff * mask

        normalized_data = {
            "mask": mask,
            "note_number": note_number,
            "f0_shifts": f0_shifts,
            "h_freq_shifts": h_freq_shifts,
            "mag_env": mag_env,
            "h_mag_dist": h_mag_dist,
            "h_phase_diff": h_phase_diff,
        }

        return normalized_data

    def denormalize(self, normalized_data, phase_mode='generated'):
        assert phase_mode in ["none", "generated", "complete"]

        mask = normalized_data["mask"]
        note_number = normalized_data["note_number"]
        f0_shifts = normalized_data["f0_shifts"]
        h_freq_shifts = normalized_data["h_freq_shifts"]
        mag_env = normalized_data["mag_env"]
        h_mag_dist = normalized_data["h_mag_dist"]

        note_number = tf.reshape(note_number, shape=(-1, 1, 1))
        note_number = tf.cast(note_number, dtype=tf.float32)
        f0_note = tsms.core.midi_to_hz(note_number)
        max_f0_displ = f0_note * self._f0_st_factor

        harmonics = tf.shape(h_freq_shifts)[2]
        harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
        harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

        # compute h_freq
        h_freq_shifts = h_freq_shifts * mask
        f0 = f0_note + f0_shifts * max_f0_displ
        h_freq = harmonic_numbers * (f0 + h_freq_shifts * max_f0_displ)

        # compute h_mag
        h_mag_dist = h_mag_dist * mask
        h_mag = self.combine_mag(mag_env, h_mag_dist)

        # compute h_phase
        h_phase = None

        if phase_mode == 'generated' or phase_mode == 'complete':
            h_phase = tsms.core.generate_phase(
                h_freq,
                sample_rate=self._sample_rate,
                frame_step=self._frame_step)

        if phase_mode == 'complete':
            h_phase_diff = normalized_data["h_phase_diff"]
            h_phase_diff = h_phase_diff * 2.0 * np.pi * mask
            h_phase = (h_phase + h_phase_diff) % (2.0 * np.pi)

        # remove zero-padding
        harmonics = tf.cast(tf.math.reduce_sum(mask[0, 0, :]), dtype=tf.int64)
        h_freq = h_freq[:, :, :harmonics]
        h_mag = h_mag[:, :, :harmonics]
        h_phase = h_phase[:, :, :harmonics] if h_phase is not None else h_phase

        return h_freq, h_mag, h_phase

    def compute_measures(self, h_freq, h_mag):
        inharmonic = heuristics.core.inharmonicity_measure(
            h_freq, h_mag, None, None, None, None)
        even_odd = heuristics.core.even_odd_measure(
            h_freq, h_mag, None, None, None, None)
        sparse_rich = heuristics.core.sparse_rich_measure(
            h_freq, h_mag, None, None, None, None)
        attack_rms = heuristics.core.attack_rms_measure(
            h_freq, h_mag, None, None, None, None)
        decay_rms = heuristics.core.decay_rms_measure(
            h_freq, h_mag, None, None, None, None)
        attack_time = heuristics.core.attack_time_measure(
            h_freq, h_mag, None, None, None, None)
        decay_time = heuristics.core.decay_time_measure(
            h_freq, h_mag, None, None, None, None)

        bass = heuristics.core.frequency_band_measure(
            h_freq, h_mag, None, None, self._sample_rate, None,
            f_min=self.freq_bands["bass"][0], f_max=self.freq_bands["bass"][1]
        )
        mid = heuristics.core.frequency_band_measure(
            h_freq, h_mag, None, None, self._sample_rate, None,
            f_min=self.freq_bands["mid"][0], f_max=self.freq_bands["mid"][1]
        )
        high_mid = heuristics.core.frequency_band_measure(
            h_freq, h_mag, None, None, self._sample_rate, None,
            f_min=self.freq_bands["high_mid"][0], f_max=self.freq_bands["high_mid"][1]
        )
        high = heuristics.core.frequency_band_measure(
            h_freq, h_mag, None, None, self._sample_rate, None,
            f_min=self.freq_bands["high"][0], f_max=self.freq_bands["high"][1]
        )

        measures = {
            "inharmonic": inharmonic,
            "even_odd": even_odd,
            "sparse_rich": sparse_rich,
            "attack_rms": attack_rms,
            "decay_rms": decay_rms,
            "attack_time": attack_time,
            "decay_time": decay_time,
            "bass": bass,
            "mid": mid,
            "high_mid": high_mid,
            "high": high,
        }

        return measures

    def output_transform(self,
                         normalized_data_true,
                         normalized_data_pred,
                         loss_data=False):
        frames = self._frames

        for k, v in self._outputs.items():
            normalized_data_pred[k] = \
                normalized_data_pred[k][:, :frames, :v["size"], ...]

            if k == "mag_env" or k == "h_mag_dist":
                if self._mag_scale_fn is not None and \
                        self._mag_loss_type != 'cross_entropy':
                    normalized_data_pred[k] = exp_sigmoid(
                        normalized_data_pred[k])

            if self._mag_loss_type == 'cross_entropy' and not loss_data:
                if k == "mag_env" or k == "h_mag_dist":
                    range_0_1 = True
                else:
                    range_0_1 = False

                normalized_data_pred[k] = tf.nn.softmax(normalized_data_pred[k])
                normalized_data_pred[k] = tf.math.argmax(
                    normalized_data_pred[k], axis=-1, output_type=tf.int32)
                normalized_data_pred[k] = mu_law_decode(
                    normalized_data_pred[k],
                    self.quantization_channels,
                    range_0_1)

        # merge dictionary to use true values where pred output is not available
        for k, v in normalized_data_true.items():
            if k not in normalized_data_pred:
                normalized_data_pred[k] = v

        return normalized_data_pred

    def freq_loss(self, y_true, y_pred, weights):
        loss = 0.0

        if self._freq_loss_type == 'mse':
            loss = tf.square(y_true - y_pred)
        elif self._freq_loss_type == 'cross_entropy':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                mu_law_encode(y_true, self.quantization_channels, False),
                y_pred)

        loss = tf.math.reduce_sum(loss * weights) / tf.math.reduce_sum(weights)
        return loss

    def mag_loss(self, y_true, y_pred, weights):
        loss = 0.0

        if self._mag_loss_type == 'mse':
            loss = tf.math.square(y_true - y_pred)
        elif self._mag_loss_type == 'rms_db':
            loss = linear_to_normalized_db(tf.math.abs(y_true - y_pred))
        elif self._mag_loss_type == 'l1_db':
            loss = tf.math.abs(
                linear_to_normalized_db(y_true, self._db_limit) -
                linear_to_normalized_db(y_pred, self._db_limit))
        elif self._mag_loss_type == 'l2_db':
            loss = tf.math.square(
                linear_to_normalized_db(y_true, self._db_limit) -
                linear_to_normalized_db(y_pred, self._db_limit))
        elif self._mag_loss_type == 'cross_entropy':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                mu_law_encode(y_true, self.quantization_channels, True),
                y_pred)

        loss = tf.math.reduce_sum(loss * weights) / tf.math.reduce_sum(weights)
        return loss

    @staticmethod
    def phase_loss(y_true, y_pred, weights):
        loss = tf.square(y_true - y_pred)
        loss = tf.math.reduce_sum(loss * weights) / tf.math.reduce_sum(weights)
        return loss

    @staticmethod
    def measures_loss(y_true, y_pred):
        loss = tf.square(y_true - y_pred)
        loss = tf.math.reduce_mean(loss)
        return loss

    def compute_mag_weight(self, mag, mask):
        weight = mask
        if self._weight_type == 'mag':
            weight = weight * mag
        elif self._weight_type == 'mag_max_pool':
            weight = weight * tf.nn.max_pool1d(
                mag, [1, self._max_pool_length, 1], 1, padding='SAME')
        return weight

    def loss(self, data_true, data_pred):
        mask = data_true["mask"]

        data_pred = self.output_transform(
            data_true, data_pred, loss_data=True)

        # compute weights
        mag_env_true = data_true["mag_env"]
        h_mag_dist_true = data_true["h_mag_dist"]

        h_mag_true = self.combine_mag(mag_env_true, h_mag_dist_true)

        env_mask = tf.ones_like(mag_env_true)
        mag_env_weight = self.compute_mag_weight(mag_env_true, env_mask)
        h_mag_weight = self.compute_mag_weight(h_mag_true, mask)

        weights = {
            "f0_shifts": mag_env_weight,
            "h_freq_shifts": h_mag_weight,
            "mag_env": env_mask,
            "h_mag_dist": h_mag_weight,
            "h_mag": mask,
            "h_phase_diff": h_mag_weight,
        }

        losses = {"loss": 0.0}

        for k, v in self._losses_weights.items():
            if v > 0.0:
                loss = 0.0
                if k == "f0_shifts" or k == "h_freq_shifts":
                    loss = self.freq_loss(
                        data_true[k],
                        data_pred[k],
                        weights[k])
                elif k == "mag_env" or k == "h_mag_dist":
                    loss = self.mag_loss(
                        data_true[k],
                        data_pred[k],
                        weights[k])
                elif k == "h_mag" and self._mag_loss_type != 'cross_entropy':
                    h_mag_pred = self.combine_mag(
                        data_pred["mag_env"],
                        data_pred["h_mag_dist"])
                    loss = self.mag_loss(h_mag_true, h_mag_pred, mask)
                elif k == "h_phase_diff":
                    loss = self.phase_loss(
                        data_true[k],
                        data_pred[k],
                        weights[k])
                elif k == "measures":
                    h_freq, h_mag, _ = self.denormalize(
                        data_pred, phase_mode='none')
                    measures_pred = self.compute_measures(h_freq, h_mag)

                    losses[k + "_loss"] = 0.0  # initialize to set position
                    for km, vm in self.measures_losses_weights.items():
                        measure_loss = vm * self.measures_loss(
                            data_true[km],
                            measures_pred[km])

                        if not self.compact_measures_logs:
                            losses[km + "_loss"] = measure_loss

                        loss += measure_loss

                loss *= v
                losses["loss"] += loss
                losses[k + "_loss"] = loss

        return losses
