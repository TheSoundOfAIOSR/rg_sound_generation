import matplotlib.pyplot as plt
import numpy as np
import ddsp
from config import *


def show_predictions(batch, encoder, decoder):
    plt.figure(figsize=(12, 12))

    f0_scaled = batch["f0_scaled"]
    note_number = batch["note_number"]
    instrument_id = batch["instrument_id"]
    sample, _, _ = encoder.predict(f0_scaled)
    pred = np.squeeze(decoder.predict([sample, note_number, instrument_id]))

    # Looking at first 4 classes
    for i in range(0, 4):
        pitch = np.argmax(note_number[i]).astype("uint8")
        pitch_to_unit = ddsp.core.midi_to_unit(pitch, midi_min=0, midi_max=127)

        plt.subplot(4, 1, 1 + i)

        plt.plot(pred[i], label="Predicted")
        plt.plot(np.squeeze(f0_scaled[i].numpy()), label="GT")
        plt.plot([pitch_to_unit] * len(pred[i]), "--")
        plt.ylim([0., 1.])
        plt.xlabel(f"pitch {pitch}")
        plt.legend()
    plt.show()
