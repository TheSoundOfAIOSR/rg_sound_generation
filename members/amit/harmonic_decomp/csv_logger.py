from typing import List
from matplotlib import pyplot as plt
from localconfig import LocalConfig


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


def show_logs(set_name: str, conf: LocalConfig):
    with open(f"{set_name}_{conf.csv_log_file}", "r") as f:
        rows = f.read().splitlines()
        losses = []
        for row in rows:
            epoch, step, loss = row.split(",")
            epoch, step, loss = int(epoch), int(step), float(loss)
            losses.append(loss)

        for e in range(0, epoch + 1):
            plt.plot(losses, label=set_name)
            plt.show()


if __name__ == "__main__":
    conf = LocalConfig()
    show_logs("train", conf)
    show_logs("val", conf)
