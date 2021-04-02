import tensorflow as tf
import os
import click
import data
import models


@click.command()
@click.option('--batch_size', default=8)
@click.option('--latent_dim', default=16)
@click.option('--learning_rate', default=float(2e-5))
def train(
        dataset_dir='d:/soundofai/complete_data',
        model_path='best.h5',
        batch_size=8,
        latent_dim=16,
        epochs=1000,
        resume=False,
        learning_rate=2e-5
):
    train_data_generator = data.data_generator(dataset_dir, 'train', batch_size)
    valid_data_generator = data.data_generator(dataset_dir, 'valid', batch_size)

    encoder = models.create_encoder(latent_dim, num_features=16)
    decoder = models.create_decoder(latent_dim, num_features=16)

    vae = models.VAE(encoder, decoder)

    # vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    vae.compile(optimizer='adam')

    if resume and os.path.isfile(model_path):
        print('Resuming training from previous save..')
        _ = vae(next(train_data_generator))
        vae.load_weights(model_path)

    steps = 200 # int(32690 / batch_size)
    validation_steps = 50 # int(2090 / batch_size)

    _ = vae.fit(
        train_data_generator,
        validation_data=valid_data_generator,
        steps_per_epoch=steps,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=2e-9,
                                                 factor=0.5, verbose=True)
        ]
    )


if __name__ == '__main__':
    train()
