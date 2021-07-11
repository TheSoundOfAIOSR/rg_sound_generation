import os
import tensorflow as tf
from loguru import logger
from .dataset import get_dataset
from . import model
from .losses import total_loss
from .localconfig import LocalConfig
from .csv_logger import write_log


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(str(len(gpus)), "Physical GPUs,", str(len(logical_gpus)), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.error(e)


def train(conf: LocalConfig):
    if conf.use_encoder:
        model_ = model.create_vae(conf)
    else:
        if conf.decoder_type == "rnn":
            model_ = model.create_rnn_decoder(conf)
        else:
            model_ = model.create_decoder(conf)

    checkpoint_path = os.path.join(conf.checkpoints_dir, f"{conf.model_name}.h5")

    if os.path.isfile(checkpoint_path):
        logger.info("Loading model_")
        model_.load_weights(checkpoint_path)
    else:
        logger.info(f"No previous checkpoint found at {checkpoint_path}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=conf.learning_rate)

    best_loss = conf.best_loss
    last_good_epoch = 0

    logger.info("Loading datasets..")
    train_dataset, valid_dataset, test_dataset = get_dataset(conf)
    logger.info("Datasets loaded")

    for epoch in range(0, conf.epochs):
        logger.info(f"Epoch {epoch} started")
        if epoch >= conf.kl_anneal_start:
            conf.kl_weight += conf.kl_anneal_factor
            conf.kl_weight = min(conf.kl_weight_max, conf.kl_weight)
        logger.info(f"Current KL weight at {conf.kl_weight}")

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
                if conf.use_encoder:
                    reconstruction, z_mean, z_log_variance = model_([h, note_number, instrument_id, velocity])
                else:
                    reconstruction = model_([note_number, velocity, instrument_id])
                    z_mean, z_log_variance = None, None
                reconstruction_loss, kl_loss = total_loss(
                    h, reconstruction, mask,
                    z_mean, z_log_variance, conf
                )

                loss = reconstruction_loss + kl_loss

            model__grads = tape.gradient(loss, model_.trainable_weights)
            model__grads, _ = tf.clip_by_global_norm(model__grads, conf.gradient_norm)

            del tape

            optimizer.apply_gradients(zip(model__grads, model_.trainable_weights))

            step_loss = loss.numpy().mean()
            losses.append(step_loss)

            if step % conf.step_log_interval == 0 and conf.log_steps:
                logger.info(f"Training Step: {step:5d}, Loss: {step_loss}, "
                      f"KL Loss: {kl_loss.numpy()}, "
                      f"R Loss: {reconstruction_loss.numpy()}")

        train_loss = sum(losses) / len(losses)

        logger.info(f"Epoch: {epoch} ended")
        logger.info(f"Training loss: {train_loss:.4f}")
        logger.info("Starting validation..")

        for valid_step, batch in enumerate(valid):
            h = batch["h"]
            h_mag = batch["h_mag"]
            mask = batch["mask"]
            note_number = batch["note_number"]
            velocity= batch["velocity"]
            instrument_id = batch["instrument_id"]

            if conf.use_encoder:
                reconstruction, z_mean, z_log_variance = model_([h, note_number, instrument_id, velocity])
            else:
                reconstruction = model_([note_number, velocity, instrument_id])
                z_mean, z_log_variance = None, None
            reconstruction_loss, kl_loss = total_loss(
                h, reconstruction, mask, h_mag,
                z_mean, z_log_variance, conf
            )

            loss = reconstruction_loss + kl_loss

            step_loss = loss.numpy().mean()
            val_losses.append(step_loss)

            if valid_step % conf.step_log_interval == 0 and conf.log_steps:
                logger.info(f"Validation Step: {valid_step:5d}, Loss: {step_loss}, "
                            f"KL Loss: {kl_loss.numpy()}, "
                            f"R Loss: {reconstruction_loss.numpy()}")

        valid_loss = sum(val_losses) / len(val_losses)
        logger.info(f"Validation Loss: {valid_loss:.4f}")

        write_log(conf, epoch, losses, val_losses)

        if valid_loss < best_loss:
            last_good_epoch = epoch
            best_loss = valid_loss
            logger.info(f"Best loss updated to {best_loss: .4f}, saving model weights")
            model_path = os.path.join(conf.checkpoints_dir,
                                      f"{epoch}_{conf.model_name}_{best_loss:.4}.h5")
            model_.save(model_path)
            logger.info(f"Updated model weights saved at {model_path}")
            conf.best_model_path = model_path
        else:
            logger.info("Validation loss did not improve")

        if epoch - last_good_epoch >= conf.early_stopping:
            logger.info(f"No improvement for {conf.early_stopping} epochs. Stopping early..")
            break

        if epoch - last_good_epoch >= conf.lr_plateau:
            logger.info(f"No improvement for {conf.lr_plateau} epochs. Reducing learning rate")
            conf.learning_rate *= conf.lr_factor
            logger.info(f"New learning rate is {conf.learning_rate}")

    if conf.best_model_path is not None:
        logger.info(f"Best model: {conf.best_model_path}")
    logger.info("Training finished")


if __name__ == "__main__":
    conf = LocalConfig()
    train(conf)
