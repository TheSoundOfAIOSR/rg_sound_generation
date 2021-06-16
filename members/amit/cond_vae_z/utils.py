import matplotlib.pyplot as plt
import numpy as np
from config import *


def show_predictions(batch, encoder, decoder, save=True):
    z_sequence = batch["z_sequence"]
    note_number = batch["note_number"]
    instrument_id = batch["instrument_id"]
    sample, _, _ = encoder.predict(z_sequence)
    pred = np.squeeze(decoder.predict([sample, note_number, instrument_id]))

    for i in range(0, 4):
        plt.close()
        plt.figure(figsize=(12, 12))

        plt.subplot(2, 1, 1)
        plt.plot(pred[i, ...], label="Predicted")
        plt.ylim([-1, 1])

        plt.subplot(2, 1, 2)
        plt.plot(z_sequence[i, ...], label="GT")
        plt.ylim([-1, 1])

        if save:
            plt.savefig(f"prediction_{i}.png")
        else:
            plt.show()
