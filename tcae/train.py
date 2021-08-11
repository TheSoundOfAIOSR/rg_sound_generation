import os
import warnings
import tensorflow as tf
from tensorflow import TensorShape
from .dataset import get_dataset
from . import model
from .losses import reconstruction_loss, kl_loss
from .localconfig import LocalConfig


warnings.simplefilter("ignore")


def _zero_batch(conf: LocalConfig):
    mask = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_harmonics])
    note_number = TensorShape([conf.batch_size, conf.num_pitches])
    velocity = TensorShape([conf.batch_size, conf.num_velocities])
    measures = TensorShape([conf.batch_size, conf.num_measures])
    f0_shifts = TensorShape([conf.batch_size, conf.harmonic_frame_steps, 1])
    mag_env = TensorShape([conf.batch_size, conf.harmonic_frame_steps, 1])
    h_freq_shifts = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_harmonics])
    h_mag_dist = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_harmonics])
    h_phase_diff = TensorShape([conf.batch_size, conf.harmonic_frame_steps, conf.max_harmonics])

    _shapes = {}

    _shapes.update({
        "mask": tf.zeros(mask),
        "f0_shifts": tf.zeros(f0_shifts),
        "mag_env": tf.zeros(mag_env),
        "h_freq_shifts": tf.zeros(h_freq_shifts),
        "h_mag_dist": tf.zeros(h_mag_dist),
        "h_phase_diff": tf.zeros(h_phase_diff)
    })

    if conf.use_note_number:
        _shapes.update({"note_number": tf.zeros(note_number)})
    if conf.use_velocity:
        _shapes.update({"velocity": tf.zeros(velocity)})
    if conf.use_heuristics:
        _shapes.update({"measures": tf.zeros(measures)})
    return _shapes


def get_zero_batch(conf: LocalConfig):
    return _zero_batch(conf)


def _step(_model, inputs, training=False):
    outputs = _model(inputs, training=training)
    losses = reconstruction_loss(inputs, outputs)

    return losses


@tf.function
def validation_step(_model, batch):
    losses = _step(_model, batch, training=False)
    return losses


@tf.function
def training_step(_model, optimizer, batch):

    with tf.GradientTape() as tape:
        losses = _step(_model, batch, training=True)

    _model_grads = tape.gradient(losses["loss"], _model.trainable_weights)
    optimizer.apply_gradients(zip(_model_grads, _model.trainable_weights))
    return losses


def dataset_loop(dataset_iterator, step_func, conf,
                 epoch, num_steps, dataset_split="Train"):
    steps = 0.0
    losses = None
    log_string = ""

    for step, batch in enumerate(dataset_iterator):
        step_losses = step_func(batch)

        steps += 1.0
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
        log_string += f"{k}: {losses[k]:.4f}, "
    log_string = log_string[:-2]

    return losses, log_string


def write_step(name, epoch, step, conf, step_losses):
    write_string = f"{epoch},{step},"
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
    if conf.use_kl_anneal:
        print(f"KL weight = {conf.kl_weight}")
        if epoch >= conf.kl_anneal_start:
            conf.kl_weight += conf.kl_anneal_factor
            conf.kl_weight = min(conf.kl_weight, conf.kl_weight_max)
            print(f"KL Weight updated to {conf.kl_weight}")


def train(conf: LocalConfig, unfrozen_layers=None):

    tf.keras.backend.clear_session()

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
        _ = _model(_zero_batch(conf))
        _model.load_weights(conf.pretrained_model_path)

        if unfrozen_layers is not None:
            print("Freezing everything except given layers")
            decoder_layer_names = [layer.name for layer in _model.decoder.layers]

            for layer in unfrozen_layers:
                assert layer in decoder_layer_names, f"{layer} to unfreeze not found in decoder"

            for layer in _model.decoder.layers:
                if layer.name not in unfrozen_layers:
                    layer.trainable = False
                else:
                    layer.trainable = True

            _model.encoder.trainable = False
            print("Model summary after freezing model:")
            print(_model.summary())
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
            return training_step(_model, optimizer, batch)

        losses, log_string = dataset_loop(
            train_iterator, train_step, conf,
            epoch, conf.num_train_steps, dataset_split="Train")

        print(log_string)

        print("Starting validation")

        def valid_step(batch):
            return validation_step(_model, batch)

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
            _model.save_weights(model_path)
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
