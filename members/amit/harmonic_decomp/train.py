import os
import tensorflow as tf
from dataset import get_dataset
from model import create_encoder, create_decoder
from localconfig import LocalConfig
from csv_logger import write_log


def train(conf: LocalConfig):
    encoder = create_encoder(conf)
    decoder = create_decoder(conf)

    if os.path.isfile("encoder.h5"):
        print("Loading encoder")
        encoder.load_weights("encoder.h5")
    if os.path.isfile("decoder.h5"):
        print("Loading decoder")
        decoder.load_weights("decoder.h5")

    optimizer = tf.keras.optimizers.Adam(learning_rate=conf.learning_rate)

    last_good_epoch = 0

    print("="*20, "Encoder", "="*20)
    print(encoder.summary())
    print("="*20, "Decoder", "="*20)
    print(decoder.summary())

    print("Loading datasets..")
    train_dataset, valid_dataset, test_dataset = get_dataset(conf)
    print("Datasets loaded")


    for epoch in range(0, conf.epochs):
        print(f"Epoch {epoch} started")
        print()
        losses = []
        val_losses = []

        train = iter(train_dataset)
        valid = iter(valid_dataset)

        for step, batch in enumerate(train):
            h = batch["h"]
            mask = batch["mask"]
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.concat([mask, mask, mask], axis=-1)
            note_number = batch["note_number"]
            velocity= batch["velocity"]
            instrument_id = batch["instrument_id"]

            with tf.GradientTape(persistent=True) as tape:
                z, z_mean, z_log_variance = encoder(h)
                reconstruction = decoder([z, note_number, velocity, instrument_id])

                reconstruction_loss = tf.sqrt(tf.square(reconstruction - h))
                reconstruction_loss = tf.math.multiply(reconstruction_loss, mask) / tf.reduce_sum(mask)
                reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, axis=(1, 2, 3)))
                kl_loss = -0.5 * (1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

                loss = reconstruction_loss + kl_loss

            encoder_grads = tape.gradient(loss, encoder.trainable_weights)
            decoder_grads = tape.gradient(loss, decoder.trainable_weights)

            encoder_grads, _ = tf.clip_by_global_norm(encoder_grads, conf.gradient_norm)
            decoder_grads, _ = tf.clip_by_global_norm(decoder_grads, conf.gradient_norm)

            del tape

            optimizer.apply_gradients(zip(encoder_grads, encoder.trainable_weights))
            optimizer.apply_gradients(zip(decoder_grads, decoder.trainable_weights))

            step_loss = loss.numpy().mean()
            losses.append(step_loss)

            if step % conf.step_log_interval == 0 and conf.log_steps:
                print(f"Step: {step:5d}, Loss at current step: {step_loss:.4f}")

        train_loss = sum(losses) / len(losses)

        print()
        print(f"Epoch: {epoch} ended")
        print(f"Training loss: {train_loss:.4f}")
        print("Starting validation..")

        for valid_step, batch in enumerate(valid):
            h = batch["h"]
            mask = batch["mask"]
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.concat([mask, mask, mask], axis=-1)
            note_number = batch["note_number"]
            velocity= batch["velocity"]
            instrument_id = batch["instrument_id"]

            z, z_mean, z_log_variance = encoder.predict(h)
            reconstruction = decoder.predict([z, note_number, velocity, instrument_id])

            reconstruction_loss = tf.sqrt(tf.square(reconstruction - h))
            reconstruction_loss = tf.math.multiply(reconstruction_loss, mask) / tf.reduce_sum(mask)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(reconstruction_loss, axis=(1, 2, 3)))
            kl_loss = -0.5 * (1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            loss = reconstruction_loss + kl_loss
            step_loss = loss.numpy().mean()

            val_losses.append(step_loss)
        valid_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {valid_loss:.4f}")

        write_log(conf, epoch, losses, val_losses)

        if valid_loss < conf.best_loss:
            last_good_epoch = epoch
            best_loss = valid_loss
            print(f"Best loss updated to {best_loss: .4f}, saving model weights")
            encoder.save("encoder.h5")
            decoder.save("decoder.h5")

        if epoch - last_good_epoch >= conf.early_stopping:
            print(f"No improvement for {conf.early_stopping} epochs. Stopping early..")
            break


if __name__ == "__main__":
    conf = LocalConfig()
    train(conf)
