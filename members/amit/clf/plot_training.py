import os
import pandas as pd
import matplotlib.pyplot as plt


for f in [x for x in os.listdir("logs") if x.lower().endswith(".csv")]:
    file_path = os.path.join("logs", f)

    df = pd.read_csv(file_path)
    epochs = df.loc[:, "epoch"].values
    accs = df.loc[:, "accuracy"].values
    val_accs = df.loc[:, "val_accuracy"].values

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accs, label="Training")
    plt.plot(epochs, val_accs, label="Validation")
    plt.ylim([0, 1])
    plt.title(f"{os.path.splitext(f)[0]}")
    plt.legend()

    plt.savefig(f"plots/{f}.png")
