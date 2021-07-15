import sys
import pandas as pd
import matplotlib.pyplot as plt


cols = ["epoch", "step", "loss", "f0_loss", "mag_env_loss",
        "h_freq_shifts_loss", "h_mag_loss", "kl_loss"]


def plot(file_path):
    df = pd.read_csv(file_path, names=cols)
    max_steps = max(df["step"].values)
    max_epochs = max(df["epoch"].values)
    losses = cols[2:]

    all_losses = {}

    for j, name in enumerate(losses):
        losses = []

        for i in range(0, max_epochs):
            loss = sum(df.loc[df["epoch"] == i][name].values)
            losses.append(loss / max_steps)

        all_losses[name] = losses
        plt.figure()
        plt.plot(losses)
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.title(name)

    plt.figure()
    for name, losses in all_losses.items():
        plt.plot(losses, label=name)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Log file path needed as first argument"
    plot(sys.argv[1])
