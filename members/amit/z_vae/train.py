import tensorflow as tf
import data
import models


def train(
        dataset_dir='d:/soundofai/complete_data',
        model_path='best.h5',
        batch_size=8,
        latent_dim=16,
        epochs=1000
):
    train_data_generator = data.data_generator(dataset_dir, 'train', batch_size)
    valid_data_generator = data.data_generator(dataset_dir, 'valid', batch_size)

    vae = models.create_vae(latent_dim, 16)
    vae.compile(
        loss='mse',
        optimizer='adam'
    )

    steps = int(32690 / batch_size)
    validation_steps = int(2090 / batch_size)

    _ = vae.fit(
        train_data_generator,
        validation_data=valid_data_generator,
        steps_per_epoch=steps,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=9),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=True)
        ]
    )


if __name__ == '__main__':
    train()
