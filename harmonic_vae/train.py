import os
import tensorflow as tf
from dataset import get_dataset
from model import create_vae
from losses import harmonic_loss
from localconfig import LocalConfig
from csv_logger import write_log


def train(conf: LocalConfig):
    vae = create_vae(conf)

    if os.path.isfile("VAE.h5"):
        print("Loading VAE")
        vae.load_weights("VAE.h5")

    optimizer = tf.keras.optimizers.Adam(learning_rate=conf.learning_rate)

    best_loss = conf.best_loss
    last_good_epoch = 0

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
            note_number = batch["note_number"]
            velocity= batch["velocity"]
            instrument_id = batch["instrument_id"]

            with tf.GradientTape(persistent=True) as tape:
                reconstruction, z_mean, z_log_variance = vae([h, note_number, instrument_id, velocity])
                reconstruction_loss = harmonic_loss(h, reconstruction, mask) / conf.batch_size

                kl_loss = -0.5 * (1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance))
                kl_loss = tf.reduce_mean(kl_loss) * conf.kl_weight

                loss = reconstruction_loss + kl_loss

            vae_grads = tape.gradient(loss, vae.trainable_weights)
            vae_grads, _ = tf.clip_by_global_norm(vae_grads, conf.gradient_norm)

            del tape

            optimizer.apply_gradients(zip(vae_grads, vae.trainable_weights))

            step_loss = loss.numpy().mean()
            losses.append(step_loss)

            if step % conf.step_log_interval == 0 and conf.log_steps:
                print(f"Step: {step:5d}, Loss at current step: {step_loss:.4f}, "
                      f"KL Loss: {kl_loss.numpy():.4f}, "
                      f"Reconstruction Loss: {reconstruction_loss.numpy():.4f}")

        train_loss = sum(losses) / len(losses)

        print()
        print(f"Epoch: {epoch} ended")
        print(f"Training loss: {train_loss:.4f}")
        print("Starting validation..")

        for valid_step, batch in enumerate(valid):
            h = batch["h"]
            mask = batch["mask"]
            note_number = batch["note_number"]
            velocity= batch["velocity"]
            instrument_id = batch["instrument_id"]

            reconstruction, z_mean, z_log_variance = vae.predict([h, note_number, instrument_id, velocity])
            reconstruction_loss = harmonic_loss(h, reconstruction, mask)

            kl_loss = -0.5 * (1 + z_log_variance - tf.square(z_mean) - tf.exp(z_log_variance))
            kl_loss = tf.reduce_mean(kl_loss) * conf.kl_weight

            loss = reconstruction_loss + kl_loss

            step_loss = loss.numpy().mean()
            val_losses.append(step_loss)

        valid_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {valid_loss:.4f}")

        write_log(conf, epoch, losses, val_losses)

        if valid_loss < best_loss:
            last_good_epoch = epoch
            best_loss = valid_loss
            print(f"Best loss updated to {best_loss: .4f}, saving model weights")
            vae.save("VAE.h5")

        if epoch - last_good_epoch >= conf.early_stopping:
            print(f"No improvement for {conf.early_stopping} epochs. Stopping early..")
            break


if __name__ == "__main__":
    conf = LocalConfig()
    train(conf)
