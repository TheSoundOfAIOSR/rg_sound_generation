import tensorflow as tf
from .localconfig import LocalConfig


# def harmonic_loss(h, reconstruction, mask):
#     f_pred, m_pred = tf.unstack(reconstruction, axis=-1)
#     f_gt, m_gt = tf.unstack(h, axis=-1)
#     m_loss = tf.square(m_pred - m_gt)
#     f_loss = tf.math.multiply(tf.square(f_pred - f_gt), m_gt)
#     reconstruction_loss = m_loss + f_loss * LocalConfig().freq_loss_weight
#     reconstruction_loss = tf.math.multiply(reconstruction_loss, mask)
#     reconstruction_loss = tf.reduce_sum(reconstruction_loss) / tf.reduce_sum(mask)
#     return reconstruction_loss

# def harmonic_loss(h, reconstruction, mask, h_mag):
#     f_pred, m_pred = tf.unstack(reconstruction, axis=-1)
#     f_gt, m_gt = tf.unstack(h, axis=-1)
#     m_loss = tf.reduce_sum(tf.square(m_pred - m_gt) * mask) / tf.reduce_sum(mask)
#     f_loss = tf.reduce_sum(tf.square(f_pred - f_gt) * (m_gt * mask)) / tf.reduce_sum(m_gt * mask)
#     reconstruction_loss = m_loss + f_loss
#     return reconstruction_loss

def harmonic_loss(h, reconstruction, mask, h_mag):
    f_pred, m_pred = tf.unstack(reconstruction, axis=-1)
    f_gt, m_gt = tf.unstack(h, axis=-1)
    m_loss = tf.reduce_sum(tf.square(m_pred - m_gt) * mask)
    f_loss = tf.reduce_sum(tf.square(f_pred - f_gt) * (h_mag * mask))
    return m_loss + f_loss
