import os
import warnings
import tensorflow as tf
import numpy as np
from .dataset import get_dataset
from . import model
from .losses import reconstruction_loss, kl_loss
from .localconfig import LocalConfig
from .compute_measures import get_measures


warnings.simplefilter("ignore")


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


def _batch(batch, conf):
    h, mask, note_number, velocity, _ = get_inputs(batch)
    all_measures = get_all_measures(batch, conf)
    return h, mask, note_number, velocity, all_measures


def _step(_model, h, mask, note_number, velocity, all_measures, conf, epoch):
    if conf.use_encoder:
        if conf.is_variational:
            reconstruction, z_mean, z_log_var = _model([h, note_number, velocity, all_measures])
        else:
            reconstruction = _model([h, note_number, velocity, all_measures])
    else:
        reconstruction = _model([note_number, velocity, all_measures])
    f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss = \
        reconstruction_loss(h, reconstruction, mask, conf)
    if conf.use_encoder and conf.is_variational:
        _kl_loss = kl_loss(z_mean, z_log_var, conf)
        if conf.use_kl_anneal:
            if epoch >= conf.kl_anneal_start:
                conf.kl_weight += conf.kl_anneal_factor
                conf.kl_weight = min(conf.kl_weight, conf.kl_weight_max)
            _kl_loss *= conf.kl_weight
            print(f"KL Weight is {conf.kl_weight:.4f} at epoch {epoch}")
    else:
        _kl_loss = 0.
    loss = f0_loss + mag_env_loss + h_freq_shifts_loss + h_mag_loss + _kl_loss
    return loss, f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss, _kl_loss


@tf.function
def validation_step(_model, batch, conf, epoch):
    h, mask, note_number, velocity, all_measures = _batch(batch, conf)
    loss, f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss, _kl_loss = \
        _step(_model, h, mask, note_number, velocity, all_measures, conf, epoch)
    return loss, f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss, _kl_loss


@tf.function
def training_step(_model, optimizer, batch, conf, epoch):
    h, mask, note_number, velocity, all_measures = _batch(batch, conf)

    with tf.GradientTape() as tape:
        loss, f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss, _kl_loss = \
            _step(_model, h, mask, note_number, velocity, all_measures, conf, epoch)
    _model_grads = tape.gradient(loss, _model.trainable_weights)
    optimizer.apply_gradients(zip(_model_grads, _model.trainable_weights))
    return loss, f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss, _kl_loss


def write_step(name, epoch, step, conf, loss,
               f0_loss, mag_env_loss, h_freq_shifts_loss,
               h_mag_loss, _kl_loss):
    write_string = f"{epoch},{step},{loss},{f0_loss},{mag_env_loss}," \
                   f"{h_freq_shifts_loss},{h_mag_loss},{_kl_loss}\n"
    out_string = f"{name}, Epoch {epoch:3d}, Step: {step:4d}, Loss: {loss:.4f} " \
                 f"F0: {f0_loss:.4f}, Mag Env: {mag_env_loss:.4f} " \
                 f"H Freq Shifts: {h_freq_shifts_loss:.4f}, H Mag: {h_mag_loss:.4f}, " \
                 f"KL Loss: {_kl_loss}"

    csv_file_path = os.path.join(conf.checkpoints_dir,
                                 f"{conf.model_name}_{name}_{conf.csv_log_file}")
    mode = "w" if epoch == 0 and step == 0 else "a"
    with open(csv_file_path, mode) as f:
        f.write(write_string)
    if step % conf.step_log_interval == 0 and conf.log_steps:
        print(out_string)


def train(conf: LocalConfig):
    print("Using configuration:")
    print("="*50)
    for key, value in vars(conf).items():
        print(f"{key} = {value}")
    print("=" * 50)
    for dh_prop in conf.data_handler_properties:
        print(f"data_handler.{dh_prop} =", eval(f"conf.data_handler.{dh_prop}"))
    print("=" * 50)
    confirm = input("Does config look ok? Y to proceed: ")
    if confirm.lower() != "y":
        print("Stopping")
        return

    print(f"Using Physical Devices:")
    print(tf.config.list_physical_devices())

    _model = model.get_model_from_config(conf)

    if conf.pretrained_model_path is not None:
        assert os.path.isfile(conf.pretrained_model_path), "No pretrained model found"
        print("Loading model")
        _model.load_weights(conf.pretrained_model_path)
    else:
        print("No pretrained model provided")

    optimizer = tf.keras.optimizers.Adam(learning_rate=conf.learning_rate)

    best_loss = conf.best_loss
    last_good_epoch = 0
    lr_changed_at = 0
    loss_names = ["loss", "f0_loss", "mag_env_loss", "h_freq_shifts_loss", "h_mag_loss", "kl_loss"]

    print("Loading datasets..")
    train_dataset, valid_dataset, test_dataset = get_dataset(conf)
    print("Datasets loaded")

    for epoch in range(0, conf.epochs):
        print(f"Epoch {epoch} started")

        losses = dict((n, []) for n in loss_names)
        val_losses = dict((n, []) for n in loss_names)

        train_set = iter(train_dataset)
        valid_set = iter(valid_dataset)

        for step, batch in enumerate(train_set):
            loss, f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss, _kl_loss = \
                training_step(_model, optimizer, batch, conf, epoch)

            losses["loss"].append(loss)
            losses["f0_loss"].append(f0_loss)
            losses["mag_env_loss"].append(mag_env_loss)
            losses["h_freq_shifts_loss"].append(h_freq_shifts_loss)
            losses["h_mag_loss"].append(h_mag_loss)
            losses["kl_loss"].append(_kl_loss)

            write_step("train", epoch, step, conf, loss, f0_loss,
                       mag_env_loss, h_freq_shifts_loss, h_mag_loss, _kl_loss)

            if conf.num_train_steps is not None:
                if step >= conf.num_train_steps:
                    print("Training steps completed")
                    break

        print(f"Epoch: {epoch} ended")
        print(f"Training losses: Loss: {np.mean(losses['loss']):.4f} "
              f"F0: {np.mean(losses['f0_loss']):.4f} "
              f"Mag Env: {np.mean(losses['mag_env_loss']):.4f} "
              f"H Freq Shifts: {np.mean(losses['h_freq_shifts_loss']):.4f} "
              f"H Mag: {np.mean(losses['h_mag_loss']):.4f} "
              f"KL: {np.mean(losses['kl_loss'])} ")
        print("Starting validation")

        for valid_step, batch in enumerate(valid_set):
            loss, f0_loss, mag_env_loss, h_freq_shifts_loss, h_mag_loss, _kl_loss = \
                validation_step(_model, batch, conf, epoch)

            val_losses["loss"].append(loss)
            val_losses["f0_loss"].append(f0_loss)
            val_losses["mag_env_loss"].append(mag_env_loss)
            val_losses["h_freq_shifts_loss"].append(h_freq_shifts_loss)
            val_losses["h_mag_loss"].append(h_mag_loss)
            val_losses["kl_loss"].append(_kl_loss)

            write_step("valid", epoch, valid_step, conf, loss, f0_loss,
                       mag_env_loss, h_freq_shifts_loss, h_mag_loss, _kl_loss)

            if conf.num_valid_steps is not None:
                if valid_step >= conf.num_valid_steps:
                    print("Validation steps completed")
                    break

        print(f"Validation losses: Loss: {np.mean(val_losses['loss']):.4f} "
              f"F0: {np.mean(val_losses['f0_loss']):.4f} "
              f"Mag Env: {np.mean(val_losses['mag_env_loss']):.4f} "
              f"H Freq Shifts: {np.mean(val_losses['h_freq_shifts_loss']):.4f} "
              f"H Mag: {np.mean(val_losses['h_mag_loss']):.4f} "
              f"KL: {np.mean(val_losses['kl_loss'])} ")

        valid_loss = np.mean(val_losses['loss'])

        if valid_loss < best_loss:
            last_good_epoch = epoch
            best_loss = valid_loss
            print(f"Best loss updated to {best_loss: .4f}, saving model weights")
            model_path = os.path.join(conf.checkpoints_dir,
                                      f"{epoch}_{conf.model_name}_{best_loss:.4}.h5")
            _model.save(model_path)
            print(f"Updated model weights saved at {model_path}")
            conf.best_model_path = model_path
        else:
            print("Validation loss did not improve")

        if epoch - last_good_epoch >= conf.early_stopping:
            print(f"No improvement for {conf.early_stopping} epochs. Stopping early")
            break

        if epoch - last_good_epoch >= conf.lr_plateau and epoch - lr_changed_at >= conf.lr_plateau:
            print("Reducing learning rate")
            conf.learning_rate *= conf.lr_factor
            print(f"New learning rate is {conf.learning_rate}")
            lr_changed_at = epoch

    if conf.best_model_path is not None:
        print(f"Best model: {conf.best_model_path}")
    print("Training finished")


if __name__ == "__main__":
    conf = LocalConfig()
    train(conf)
