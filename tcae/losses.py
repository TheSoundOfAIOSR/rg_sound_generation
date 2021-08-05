import tensorflow as tf
from .localconfig import LocalConfig


def reconstruction_loss(inputs, outputs):
    conf = LocalConfig()
    mask = inputs["mask"]
    normalized_data_true = inputs
    normalized_data_pred = outputs

    normalized_data_pred = \
        conf.data_handler.prediction_transform(normalized_data_pred,
                                               loss_data=True)

    losses = conf.data_handler.loss(
        normalized_data_true, normalized_data_pred, mask)

    return losses


def kl_loss(z_mean, z_log_variance):
    conf = LocalConfig()
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
