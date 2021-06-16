import os
import tensorflow as tf
from dataset import train_dataset, valid_dataset
from model import create_encoder, create_decoder
from utils import show_predictions
from config import *


encoder = create_encoder()
decoder = create_decoder()

if os.path.isfile("encoder.h5"):
    print("Loading encoder")
    encoder.load_weights("encoder.h5")
if os.path.isfile("decoder.h5"):
    print("Loading decoder")
    decoder.load_weights("decoder.h5")

optimizer = tf.keras.optimizers.Adam()
train = iter(train_dataset)
valid = iter(valid_dataset)

best_loss = 1e6
last_good_epoch = 0


print("="*20, "Encoder", "="*20)
print(encoder.summary())
print("="*20, "Decoder", "="*20)
print(decoder.summary())

for epoch in range(0, epochs):
    print(f"Epoch {epoch} started")
    print()
    cumulative_loss = 0.
    valid_loss = 0.
    for step in range(0, steps):
        batch = next(train)
        z_sequence = batch["z_sequence"]
        note_number = batch["note_number"]
        instrument_id = batch["instrument_id"]

        with tf.GradientTape(persistent=True) as tape:
            z, z_mean, z_log_variance = encoder(z_sequence)
            reconstruction = decoder([z, note_number, instrument_id])
            reconstruction_loss = tf.keras.losses.mean_squared_error(reconstruction, z_sequence)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, axis=1))
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

        if step % 100 == 0:
            print(end="=")

    cumulative_loss = float(cumulative_loss) / steps

    print()
    print(f"Epoch: {epoch} ended")
    print(f"Training loss: {cumulative_loss:.4f}")

    # for val_step in range(0, validation_steps):
    #     batch = next(valid)
    #     z_sequence = batch["z_sequence"]
    #     note_number = batch["note_number"]
    #     instrument_id = batch["instrument_id"]
    #
    #     z, z_mean, z_log_variance = encoder.predict(z_sequence)
    #     reconstruction = decoder.predict([z, note_number, instrument_id])
    #     reconstruction_loss = tf.keras.losses.mean_squared_error(reconstruction, z_sequence)
    #     reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, axis=1))
    #     kl_loss = -0.5 * (1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance))
    #     kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    #
    #     loss = reconstruction_loss + kl_loss
    #     step_loss = loss.numpy().mean()
    #     valid_loss += step_loss
    #
    #     if val_step % 100 == 0:
    #         print(end="=")
    #
    # valid_loss = float(valid_loss) / validation_steps
    # print(f"Validation loss: {valid_loss:.4f}")

    if cumulative_loss < best_loss:
        last_good_epoch = epoch
        best_loss = cumulative_loss
        print(f"Best loss updated to {best_loss: .4f}, saving model weights")
        encoder.save("encoder.h5")
        decoder.save("decoder.h5")

    batch = next(train)
    show_predictions(batch, encoder, decoder)

    if epoch - last_good_epoch >= early_stopping:
        print(f"No improvement for {early_stopping} epochs. Stopping early..")
        break
