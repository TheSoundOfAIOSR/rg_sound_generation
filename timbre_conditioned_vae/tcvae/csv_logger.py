from typing import List
import numpy as np
from matplotlib import pyplot as plt
from .localconfig import LocalConfig


def write_log(conf: LocalConfig, epoch: int, train_losses: List[float], val_losses: List[float]):
    print("Writing logs..")
    mode = "w" if epoch == 0 else "a"

    with open(f"train_{conf.csv_log_file}", mode) as f:
        for step, l in enumerate(train_losses):
            f.write(f"{epoch},{step},{l}\n")

    with open(f"val_{conf.csv_log_file}", mode) as f:
        for step, l in enumerate(val_losses):
            f.write(f"{epoch},{step},{l}\n")

    print(f"Logs written for epoch {epoch}")


def show_logs(set_name: str, conf: LocalConfig, all_steps: bool = False):
    with open(f"{set_name}_{conf.csv_log_file}", "r") as f:
        rows = f.read().splitlines()
        epochs = []
        losses = []

        for row in rows:
            epoch, _, loss = row.split(",")
            epoch, loss = int(epoch), float(loss)
            epochs.append(epoch)
            losses.append(loss)

        if all_steps:
            plt.plot(losses, label=set_name)
            plt.legend()
            plt.show()
            return

        epochs = np.array(epochs)
        losses = np.array(losses)
        total_epochs = np.max(epochs)
        epoch_losses = []

        for e in range(0, total_epochs):
            indices = np.where(epochs == e)
            current_losses = losses[indices]
            epoch_losses.append(current_losses.mean())

        plt.plot(epoch_losses, label=set_name)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    conf = LocalConfig()
    show_logs("train", conf)
    show_logs("val", conf)
