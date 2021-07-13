import tensorflow as tf
from .localconfig import LocalConfig


def deconstruct_tensors(h, reconstruction, mask, conf):
    f_true, m_true = tf.unstack(h, axis=-1)
    f_pred, m_pred = tf.unstack(reconstruction, axis=-1)

    max_harmonics = conf.data_handler.max_harmonics

    f0_shifts_true = f_true[:, :, 0:1]
    f0_shifts_pred = f_pred[:, :, 0:1]

    mag_env_true = m_true[:, :, 0:1]
    mag_env_pred = m_pred[:, :, 0:1]

    h_freq_shifts_true = f_true[:, :, 1:max_harmonics + 1]
    h_freq_shifts_pred = f_pred[:, :, 1:max_harmonics + 1]

    h_mag_distribution_true = m_true[:, :, 1:max_harmonics + 1]
    h_mag_distribution_pred = m_pred[:, :, 1:max_harmonics + 1]

    mask = mask[:, :, 0:max_harmonics]
    return (f0_shifts_true, f0_shifts_pred, mag_env_true, mag_env_pred,
            h_freq_shifts_true, h_freq_shifts_pred, h_mag_distribution_true,
            h_mag_distribution_pred, mask)


def reconstruction_loss(h, reconstruction, mask, conf):
    (f0_shifts_true, f0_shifts_pred, mag_env_true, mag_env_pred,
     h_freq_shifts_true, h_freq_shifts_pred, h_mag_distribution_true,
     h_mag_distribution_pred, mask) = deconstruct_tensors(h, reconstruction, mask, conf)

    f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss = \
        conf.data_handler.loss(
            f0_shifts_true, f0_shifts_pred,
            mag_env_true, mag_env_pred,
            h_freq_shifts_true, h_freq_shifts_pred,
            h_mag_distribution_true, h_mag_distribution_pred,
            mask)

    f0_weight = 1.
    mag_env_weight = 1.
    h_freq_shifts_weight = 1.
    h_mag_weight = 1.

    return (f0_loss * f0_weight, mag_env_loss * mag_env_weight,
            h_freq_shifts_loss * h_freq_shifts_weight, h_mag_loss * h_mag_weight)


def kl_loss(z_mean, z_log_variance, conf):
    loss = -0.5 * (1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance))
    loss = tf.reduce_mean(loss) * conf.kl_weight
    return loss


def total_loss(h, reconstruction, mask,
               z_mean, z_log_variance, conf: LocalConfig):
    _r_loss = reconstruction_loss(h, reconstruction, mask, conf)
    if conf.use_encoder:
        _kl_loss = kl_loss(z_mean, z_log_variance, conf)
        return _r_loss, _kl_loss
    return _r_loss, tf.constant(0.)
