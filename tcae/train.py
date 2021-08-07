import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter
import os
import warnings
from tcae.dataset import get_dataset
from tcae.model import TCAEModel
from tcae.localconfig import LocalConfig

warnings.simplefilter("ignore")


class ModelWrapper(tf.keras.Model):
    def __init__(self, model, loss_fn):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        inputs, targets, sample_weight = \
            data_adapter.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            outputs = self(inputs, training=True)

            losses = self.loss_fn(targets, outputs)
            loss = losses["loss"]

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return losses

    def test_step(self, data):
        data = data_adapter.expand_1d(data)
        inputs, targets, sample_weight = \
            data_adapter.unpack_x_y_sample_weight(data)

        outputs = self(inputs, training=False)
        losses = self.loss_fn(targets, outputs)

        return losses

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    def get_config(self):
        return self.model.get_config()


def train(conf: LocalConfig):
    conf = LocalConfig() if conf is None else conf

    # create dataset
    print("Loading datasets...")
    train_dataset, valid_dataset, test_dataset = get_dataset(conf)

    # create and compile model
    model = ModelWrapper(
        TCAEModel(conf),
        conf.data_handler.loss)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=conf.learning_rate),
        run_eagerly=False)

    # build model
    x, y = next(iter(train_dataset))
    _ = model(x)

    if conf.print_model_summary:
        print(model.summary())

    # load model checkpoint
    if conf.pretrained_model_path is not None:
        if os.path.isfile(conf.pretrained_model_path):
            model.load_weights(conf.pretrained_model_path)
        else:
            print(f"No pretrained model found at {conf.pretrained_model_path}")

    # create training callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(conf.checkpoints_dir, conf.model_name + "_{epoch}_{val_loss:.5f}.ckpt"),
        monitor='val_loss',
        save_weights_only=True,
        verbose=1,
        save_best_only=True,
        save_freq='epoch')

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=conf.early_stopping, verbose=1)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=conf.lr_factor, patience=conf.lr_plateau,
        verbose=1)

    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(conf.checkpoints_dir, f"{conf.model_name}_{conf.csv_log_file}"),

    )

    # train model
    model.fit(
        train_dataset,
        epochs=conf.epochs,
        validation_data=valid_dataset,
        callbacks=[
            checkpoint, early_stop,
            reduce_lr, csv_logger
        ])

    # model.fit(
    #     train_dataset,
    #     epochs=10,
    #     steps_per_epoch=5,
    #     validation_data=valid_dataset,
    #     validation_steps=5,
    #     callbacks=[checkpoint, early_stop, reduce_lr])

    # evaluate model
    model.evaluate(test_dataset)
