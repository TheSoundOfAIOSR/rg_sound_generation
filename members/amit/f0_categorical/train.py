import click
import os
import tensorflow as tf

from data import data_generator, complete_record_generator
from model import create_model


@click.command()
@click.option('--dataset_dir')
@click.option('--model_dir')
@click.option('--batch_size', default=8)
@click.option('--epochs', default=100)
@click.option('--num_train_ex', default=32690)
@click.option('--num_valid_ex', default=2081)
def train(dataset_dir, model_dir, batch_size, epochs, num_train_ex, num_valid_ex):
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
    elif os.path.isfile(model_path):
        click.echo('Loading best weights from previous training..')
        model.load_weights(model_path)
    
    steps_per_epoch = max(1, int(num_train_ex/batch_size))
    validation_steps = max(1, int(num_valid_ex/batch_size))

    click.echo(f'steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}')

    click.echo('Starting training..')
    _ = model.fit(
        train_generator,
        validation_data=valid_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10),
            tf.keras.callbacks.ModelCheckpoint(
                model_path, monitor='val_accuracy', save_best_only=True
            )
        ],
        verbose=True
    )
    click.echo('Training finished..')


if __name__ == '__main__':
    train()
