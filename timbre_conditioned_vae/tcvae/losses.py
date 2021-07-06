import tensorflow as tf


def harmonic_loss(h, reconstruction, mask, h_mag):
    f_pred, m_pred = tf.unstack(reconstruction, axis=-1)
    f_gt, m_gt = tf.unstack(h, axis=-1)

    weighted_mask = h_mag * mask

    m_loss = tf.reduce_sum(tf.square(m_pred - m_gt) * weighted_mask)
    f_loss = tf.reduce_sum(tf.square(f_pred - f_gt) * weighted_mask)
    return m_loss + f_loss
