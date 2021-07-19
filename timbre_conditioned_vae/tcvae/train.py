import os
import warnings
import tensorflow as tf
from .dataset import get_dataset
from . import model
from .losses import reconstruction_loss, kl_loss
from .localconfig import LocalConfig


warnings.simplefilter("ignore")


def _batch(batch):
    note_number = batch["note_number"]
    velocity = batch["velocity"]
    h = batch["h"]
    mask = batch["mask"]
    measures = batch["measures"]
    return note_number, velocity, h, mask, measures


def _step(_model, h, mask, note_number, velocity, measures, conf):
    model_input = [note_number, velocity]
    if conf.use_heuristics:
        model_input += [measures]
    if conf.use_encoder:
        model_input = [h] + model_input
    if conf.use_encoder and conf.is_variational:
        reconstruction, z_mean, z_log_var = _model(model_input)
    else:
        reconstruction = _model(model_input)
    losses = reconstruction_loss(h, reconstruction, mask, conf)
    if conf.use_encoder and conf.is_variational:
        # Note this is weighted kl loss as kl_loss function applies the weight
        _kl_loss = kl_loss(z_mean, z_log_var, conf)
    else:
        _kl_loss = 0.

    losses["kl_loss"] = _kl_loss

    return losses


@tf.function
def validation_step(_model, batch, conf):
    note_number, velocity, h, mask, measures = _batch(batch)

    losses = _step(_model, h, mask, note_number, velocity, measures, conf)

    return losses


@tf.function
def training_step(_model, optimizer, batch, conf):
    note_number, velocity, h, mask, measures = _batch(batch)

    with tf.GradientTape() as tape:
        losses = _step(_model, h, mask, note_number, velocity, measures, conf)

    _model_grads = tape.gradient(losses["loss"], _model.trainable_weights)
    optimizer.apply_gradients(zip(_model_grads, _model.trainable_weights))
    return losses


def dataset_loop(dataset_iterator, step_func, conf,
                 epoch, num_steps, dataset_split="Train"):
    steps = 0
    losses = None
    log_string = ""

    for step, batch in enumerate(dataset_iterator):
        step_losses = step_func(batch)

        steps += 1
        if losses is None:
            losses = step_losses
        else:
            for k, v in step_losses.items():
                losses[k] += v

        write_step(dataset_split, epoch, step, conf, step_losses)

        if num_steps is not None:
            if step >= num_steps:
                log_string += dataset_split + " steps completed\n"
                break

    log_string = f"{dataset_split} losses: "
    for k, v in losses.items():
        losses[k] = v / steps
        log_string += f"{k}: {v:.4f}, "
    log_string = log_string[:-2]

    return losses, log_string


def write_step(name, epoch, step, conf, step_losses):
    write_string = ""
    out_string = f"{name}, Epoch {epoch:3d}, Step: {step:4d}, "
    for k, v in step_losses.items():
        write_string += f"{v},"
        out_string += f"{k}: {v:.4f}, "
    write_string = write_string[:-1] + "\n"
    out_string = out_string[:-2]

    csv_file_path = os.path.join(
        conf.checkpoints_dir, f"{conf.model_name}_{name}_{conf.csv_log_file}")
    mode = "w" if epoch == 0 and step == 0 else "a"
    with open(csv_file_path, mode) as f:
        f.write(write_string)
    if step % conf.step_log_interval == 0 and conf.log_steps:
        print(out_string)


def set_kl_weight(epoch, conf: LocalConfig):
    if not conf.use_kl_anneal:
        return
    if epoch >= conf.kl_anneal_start:
        conf.kl_weight += conf.kl_anneal_factor
        conf.kl_weight = min(conf.kl_weight, conf.kl_weight_max)
        print(f"KL Weight set to {conf.kl_weight}")


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

    print("Loading datasets..")
    train_dataset, valid_dataset, test_dataset = get_dataset(conf)
    print("Datasets loaded")

    for epoch in range(0, conf.epochs):
        print(f"Epoch {epoch} started")

        train_iterator = iter(train_dataset)
        valid_iterator = iter(valid_dataset)

        if conf.is_variational:
            set_kl_weight(epoch, conf)

        def train_step(batch):
            return training_step(_model, optimizer, batch, conf)

        losses, log_string = dataset_loop(
            train_iterator, train_step, conf,
            epoch, conf.num_train_steps, dataset_split="Train")

        print(log_string)

        print("Starting validation")

        def valid_step(batch):
            return validation_step(_model, batch, conf)

        losses, log_string = dataset_loop(
            valid_iterator, valid_step, conf,
            epoch, conf.num_valid_steps, dataset_split="Valid")

        print(log_string)

        valid_loss = losses['loss']

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
