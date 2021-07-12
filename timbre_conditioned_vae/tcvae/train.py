import os
import tensorflow as tf
from loguru import logger
from .dataset import get_dataset
from . import model
from .losses import reconstruction_loss
from .localconfig import LocalConfig
from .csv_logger import write_log
from .preprocess import get_measures


def get_inputs(batch):
    h = batch["h"]
    mask = batch["mask"]
    note_number = batch["note_number"]
    velocity = batch["velocity"]
    instrument_id = batch["instrument_id"]
    return h, mask, note_number, velocity, instrument_id


def get_all_measures(batch, conf):
    h_freq_orig = batch["h_freq_orig"]
    h_mag_orig = batch["h_mag_orig"]
    harmonics = batch["harmonics"]
    return get_measures(h_freq_orig, h_mag_orig, harmonics, conf)


def write_step(name, step, conf, step_loss):
    if step % conf.step_log_interval == 0 and conf.log_steps:
        logger.info(f"{name} Step: {step:5d}, Loss: {step_loss}")


def train(conf: LocalConfig):

    logger.info(f"Using Physical Devices:")
    print(tf.config.list_physical_devices())

    if conf.decoder_type == "rnn":
        _model = model.create_rnn_decoder(conf)
    else:
        _model = model.create_decoder(conf)

    checkpoint_path = os.path.join(conf.checkpoints_dir, f"{conf.model_name}.h5")

    if os.path.isfile(checkpoint_path):
        logger.info("Loading model")
        _model.load_weights(checkpoint_path)
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
        # if epoch >= conf.kl_anneal_start:
        #     conf.kl_weight += conf.kl_anneal_factor
        #     conf.kl_weight = min(conf.kl_weight_max, conf.kl_weight)
        # logger.info(f"Current KL weight at {conf.kl_weight}")

        losses = []
        val_losses = []

        train_set = iter(train_dataset)
        valid_set = iter(valid_dataset)

        for step, batch in enumerate(train_set):
            h, mask, note_number, velocity, instrument_id = get_inputs(batch)
            all_measures = get_all_measures(batch, conf)

            with tf.GradientTape(persistent=True) as tape:
                reconstruction = _model([note_number, velocity, all_measures])
                loss = tf.squeeze(reconstruction_loss(h, reconstruction, mask, conf))

            step_loss = loss.numpy().mean()
            losses.append(step_loss)

            write_step("Training", step, conf, step_loss)

            _model_grads = tape.gradient(loss, _model.trainable_weights)
            # _model_grads, _ = tf.clip_by_global_norm(_model_grads, conf.gradient_norm)

            del tape

            optimizer.apply_gradients(zip(_model_grads, _model.trainable_weights))

        train_loss = sum(losses) / len(losses)

        logger.info(f"Epoch: {epoch} ended")
        logger.info(f"Training loss: {train_loss:.4f}")
        logger.info("Starting validation..")

        for valid_step, batch in enumerate(valid_set):
            h, mask, note_number, velocity, instrument_id = get_inputs(batch)
            all_measures = get_all_measures(batch, conf)
            reconstruction = _model([note_number, velocity, all_measures])
            loss = tf.squeeze(reconstruction_loss(h, reconstruction, mask, conf))

            step_loss = loss.numpy().mean()
            val_losses.append(step_loss)

            write_step("Training", valid_step, conf, step_loss)

        valid_loss = sum(val_losses) / len(val_losses)
        logger.info(f"Validation Loss: {valid_loss:.4f}")

        write_log(conf, epoch, losses, val_losses)

        if valid_loss < best_loss:
            last_good_epoch = epoch
            best_loss = valid_loss
            logger.info(f"Best loss updated to {best_loss: .4f}, saving model weights")
            model_path = os.path.join(conf.checkpoints_dir,
                                      f"{epoch}_{conf.model_name}_{best_loss:.4}.h5")
            _model.save(model_path)
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
            conf.lr_plateau += 3 # Wait 3 epochs before reducing lr again

    if conf.best_model_path is not None:
        logger.info(f"Best model: {conf.best_model_path}")
    logger.info("Training finished")


if __name__ == "__main__":
    conf = LocalConfig()
    train(conf)
