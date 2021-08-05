import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import core


def main():
    envelope0 = core.dahdr_envelope(
        attack_alpha=-6, decay_alpha=-3, release_alpha=-6,
        delay_samples=20, attack_samples=40, hold_samples=0,
        decay_samples=500, release_samples=200)

    envelope1 = core.dahdr_envelope(
        attack_alpha=6, decay_alpha=3, release_alpha=-6,
        delay_samples=20, attack_samples=40, hold_samples=0,
        decay_samples=500, release_samples=50)

    envelope2 = core.dahdr_envelope(
        attack_alpha=1, decay_alpha=-20, release_alpha=0,
        delay_samples=20, attack_samples=40, hold_samples=0,
        decay_samples=500, release_samples=0)

    plt.figure()
    plt.plot(np.squeeze(envelope0.numpy()))

    plt.figure()
    plt.plot(np.squeeze(envelope1.numpy()))

    plt.figure()
    plt.plot(np.squeeze(envelope2.numpy()))

    plt.show()


if __name__ == '__main__':
    main()

