import tensorflow as tf


class SamplingLayer(tf.keras.layers.Layer):
    def call(self, inputs, *_):
        z_mean, z_log_variance = inputs
        batch_size = tf.shape(z_mean)[0]
        latent_dimension = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, latent_dimension))
        return z_mean + tf.exp(0.5 * z_log_variance) * epsilon
