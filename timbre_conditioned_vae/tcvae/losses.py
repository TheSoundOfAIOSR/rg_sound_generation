import tensorflow as tf
from .localconfig import LocalConfig


def reconstruction_loss(h, reconstruction, mask, h_mag, conf):
    f_pred, m_pred = tf.unstack(reconstruction, axis=-1)
    f_gt, m_gt = tf.unstack(h, axis=-1)
    weighted_mask = h_mag * mask
    m_loss = tf.reduce_sum(tf.square(m_pred - m_gt) * weighted_mask)
    f_loss = tf.reduce_sum(tf.square(f_pred - f_gt) * weighted_mask)
    return (m_loss + f_loss) * conf.reconstruction_weight / conf.batch_size


def kl_loss(z_mean, z_log_variance, conf):
    loss = -0.5 * (1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance))
    loss = tf.reduce_mean(loss) * conf.kl_weight
    return loss


def total_loss(h, reconstruction, mask, h_mag,
               z_mean, z_log_variance, conf: LocalConfig):
    _r_loss = reconstruction_loss(h, reconstruction, mask, h_mag, conf)
    if conf.use_encoder:
        _kl_loss = kl_loss(z_mean, z_log_variance, conf)
        return _r_loss, _kl_loss
    return _r_loss, tf.constant(0.)
