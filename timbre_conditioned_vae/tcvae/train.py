import os
import tensorflow as tf
from loguru import logger
from .dataset import get_dataset
from . import model
from .losses import reconstruction_loss
from .localconfig import LocalConfig
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


def write_step(name, epoch, step, conf, loss,
               f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss):
    write_string = f"{epoch},{step},{loss},{f0_loss},{mag_env_loss},{h_freq_shifts_loss},{h_mag_loss}\n"
    out_string = f"{name}, Epoch {epoch}, Step: {step}, Loss: {loss} " \
                 f"F0: {f0_loss}, Mag Env: {mag_env_loss} " \
                 f"H Freq Shifts: {h_freq_shifts_loss}, H Mag: {h_mag_loss}"

    with open(f"{name}_{conf.csv_log_file}", "a") as f:
        f.write(write_string)
    if step % conf.step_log_interval == 0 and conf.log_steps:
        logger.info(out_string)


def train(conf: LocalConfig):

    logger.info(f"Using Physical Devices:")
    print(tf.config.list_physical_devices())

    if os.path.isfile(f"train_{conf.csv_log_file}"):
        os.remove(f"train_{conf.csv_log_file}")
    if os.path.isfile(f"valid_{conf.csv_log_file}"):
        os.remove(f"train_{conf.csv_log_file}")

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

        losses = []
        val_losses = []

        train_set = iter(train_dataset)
        valid_set = iter(valid_dataset)

        for step, batch in enumerate(train_set):
            h, mask, note_number, velocity, instrument_id = get_inputs(batch)
            all_measures = get_all_measures(batch, conf)

            with tf.GradientTape() as tape:
                reconstruction = _model([note_number, velocity, all_measures])
                f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss = \
                    reconstruction_loss(h, reconstruction, mask, conf)
                loss = f0_loss + mag_env_loss + h_freq_shifts_loss + h_mag_loss

            step_loss = tf.squeeze(loss).numpy().mean()
            losses.append(step_loss)

            write_step("train", epoch, step, conf, step_loss, f0_loss,
                       mag_env_loss, h_freq_shifts_loss, h_mag_loss)

            _model_grads = tape.gradient(loss, _model.trainable_weights)
            optimizer.apply_gradients(zip(_model_grads, _model.trainable_weights))

        train_loss = sum(losses) / len(losses)

        logger.info(f"Epoch: {epoch} ended")
        logger.info(f"Training loss: {train_loss:.4f}")
        logger.info("Starting validation")

        for valid_step, batch in enumerate(valid_set):
            h, mask, note_number, velocity, instrument_id = get_inputs(batch)
            all_measures = get_all_measures(batch, conf)
            reconstruction = _model([note_number, velocity, all_measures])
            f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss = \
                reconstruction_loss(h, reconstruction, mask, conf)
            loss = f0_loss + mag_env_loss + h_freq_shifts_loss + h_mag_loss

            step_loss = tf.squeeze(loss).numpy().mean()
            val_losses.append(step_loss)

            write_step("valid", epoch, valid_step, conf, step_loss, f0_loss,
                       mag_env_loss, h_freq_shifts_loss, h_mag_loss)

        valid_loss = sum(val_losses) / len(val_losses)
        logger.info(f"Validation Loss: {valid_loss:.4f}")

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
            logger.info(f"No improvement for {conf.early_stopping} epochs. Stopping early")
            break

        if epoch - last_good_epoch >= conf.lr_plateau:
            logger.info(f"No improvement for {conf.lr_plateau} epochs. Reducing learning rate")
            conf.learning_rate *= conf.lr_factor
            logger.info(f"New learning rate is {conf.learning_rate}")
            conf.lr_plateau += 2 # Wait 2 epochs before reducing lr again

    if conf.best_model_path is not None:
        logger.info(f"Best model: {conf.best_model_path}")
    logger.info("Training finished")


if __name__ == "__main__":
    conf = LocalConfig()
    train(conf)
