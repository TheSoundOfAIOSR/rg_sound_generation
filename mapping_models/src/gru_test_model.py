"""
Run from command line:
    python gru_test_model.py --dataset_dir ../nsynth_dataset --model_dir gru
"""

import tensorflow as tf
import trainer
import click


model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(32, return_sequences=True),
    tf.keras.layers.Dense(2, activation='tanh')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=[tf.keras.losses.MeanSquaredError()]
)


@click.command()
@click.option('--dataset_dir', help='Location of root directory of the dataset')
@click.option('--model_dir_name',
              help='Name of checkpoint directory, will be created inside the main checkpoint directory')
def train(dataset_dir, model_dir_name):
    trainer.train(
        model,
        dataset_dir=dataset_dir,
        model_dir=model_dir_name
    )


if __name__ == '__main__':
    train()
