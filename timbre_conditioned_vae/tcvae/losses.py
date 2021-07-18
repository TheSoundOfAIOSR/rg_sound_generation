import tensorflow as tf
from .localconfig import LocalConfig


def reconstruction_loss(h_true, h_pred, mask, conf):
    f0_shifts_true, mag_env_true, \
    h_freq_shifts_true, h_mag_dist_true, h_phase_diff_true = \
        conf.data_handler.output_transform(h_true, pred=False)

    f0_shifts_pred, mag_env_pred, \
    h_freq_shifts_pred, h_mag_dist_pred, h_phase_diff_pred = \
        conf.data_handler.output_transform(h_pred, pred=True)

    f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss, h_phase_diff_loss = \
        conf.data_handler.loss(
            f0_shifts_true, f0_shifts_pred,
            mag_env_true, mag_env_pred,
            h_freq_shifts_true, h_freq_shifts_pred,
            h_mag_dist_true, h_mag_dist_pred,
            h_phase_diff_true, h_phase_diff_pred,
            mask)

    return f0_loss, mag_env_loss, \
           h_freq_shifts_loss, h_mag_loss, h_phase_diff_loss


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
