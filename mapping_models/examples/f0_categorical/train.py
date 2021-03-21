import click
import os
import tensorflow as tf

from data import data_generator, complete_record_generator
from model import create_model


@click.command('start')
@click.option('--dataset_dir')
@click.option('--model_dir')
@click.option('--batch_size', default=8)
@click.option('--epochs', default=100)
def train(dataset_dir, model_dir, batch_size, epochs):
    click.echo('Creating data generators..')
    train_generator = data_generator(complete_record_generator(
        dataset_dir=dataset_dir,
        set_name='train',
        batch_size=batch_size
    ))

    valid_generator = data_generator(complete_record_generator(
        dataset_dir=dataset_dir,
        set_name='valid',
        batch_size=batch_size
    ))

    click.echo('Creating model..')
    model = create_model()
    print(model.summary())

    model_path = os.path.join(model_dir, 'best.h5')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    click.echo('Starting training..')
    _ = model.fit(
        train_generator,
        validation_data=valid_generator,
        steps_per_epoch=int(32690/batch_size),
        validation_steps=int(2081/batch_size),
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor='val_accuracy', save_best_only=True
            )
        ],
        verbose=True
    )
    click.echo('Training finished..')


if __name__ == '__main__':
    train()
