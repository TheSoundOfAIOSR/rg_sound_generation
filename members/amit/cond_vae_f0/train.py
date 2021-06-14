import tensorflow as tf
from dataset import train_dataset, valid_dataset
from model import create_encoder, create_decoder
from utils import show_predictions
from config import *


encoder = create_encoder()
decoder = create_decoder()
optimizer = tf.keras.optimizers.RMSprop()
train = iter(train_dataset)
valid = iter(valid_dataset)


for epoch in range(0, epochs):
    cumulative_loss = 0.
    for step in range(0, steps):
        batch = next(train)
        f0_scaled = batch["f0_scaled"]
        note_number = batch["note_number"]
        instrument_id = batch["instrument_id"]

        with tf.GradientTape(persistent=True) as tape:
            z, z_mean, z_log_variance = encoder(f0_scaled)
            reconstruction = decoder([z, note_number, instrument_id])
            reconstruction_loss = tf.keras.losses.mean_squared_error(reconstruction, f0_scaled)
            reconstruction_loss = tf.reduce_mean(reconstruction_loss)
            kl_loss = -0.5 * (1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            loss = reconstruction_loss + kl_loss

        encoder_grads = tape.gradient(loss, encoder.trainable_weights)
        decoder_grads = tape.gradient(loss, decoder.trainable_weights)

        del tape

        optimizer.apply_gradients(zip(encoder_grads, encoder.trainable_weights))
        optimizer.apply_gradients(zip(decoder_grads, decoder.trainable_weights))

        step_loss = loss.numpy().mean()
        cumulative_loss += step_loss

    print(f"Epoch: {epoch:3d}, Loss: {float(cumulative_loss) / steps:.4f}")

    valid_batch = next(valid)
    show_predictions(valid_batch, encoder, decoder)
