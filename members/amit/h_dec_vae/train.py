import click
from model import build_vae
from dataset import get_data
from preprocessing import input_preprocessing


@click.command()
@click.option("--dataset_dir")
def train(dataset_dir):
    train, _, _ = get_data(dataset_dir)
    X_train = input_preprocessing(train)
    # X_valid = input_preprocessing(valid)
    # X_test = input_preprocessing(test)

    vae = build_vae(latent_dim=128, lstm_dim=128, learning_rate=0.001,
                    units=[64, 64, 128, 128, 128, 128], kernel_sizes=[7, 7, 5, 5, 5, 5], strides=[3, 3, 2, 2, 2, 2])

    batch_size = 4
    steps = int(14500/batch_size)

    vae.fit(
        X_train,
        epochs=2,
        steps_per_epoch=steps,
        batch_size=batch_size
    )


if __name__ == "__main__":
    train()
