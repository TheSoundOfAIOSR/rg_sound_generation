import tensorflow as tf
import tsms
import matplotlib.pyplot as plt
from model import create_encoder, create_decoder
from localconfig import LocalConfig
from dataset import get_dataset


conf = LocalConfig()
_, _, test_dataset = get_dataset(conf)
test_iterable = iter(test_dataset)


encoder = create_encoder(conf)
decoder = create_decoder(conf)
encoder.load_weights("encoder.h5")
decoder.load_weights("decoder.h5")


def specgrams(x):
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.specgram(x, NFFT=256, Fs=conf.sample_rate, window=None,
                  noverlap=256 - conf.frame_size, mode='psd', vmin=-180)
    plt.subplot(2, 1, 2)
    plt.specgram(x, NFFT=1024, Fs=conf.sample_rate, window=None,
                  noverlap=1024 - conf.frame_size, mode='psd', vmin=-180)
    plt.show()


def next_prediction():
    batch = next(test_iterable)
    h = batch["h"]
    mask = batch["mask"]
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.concat([mask, mask, mask], axis=-1)
    note_number = batch["note_number"]
    velocity = batch["velocity"]
    instrument_id = batch["instrument_id"]

    z, z_mean, z_log_variance = encoder.predict(h)
    reconstruction = decoder.predict([z, note_number, velocity, instrument_id])

    reconstruction = tf.math.multiply(reconstruction, mask)

    h_freq_gt = h[0, ..., 0]
    h_mag_gt = h[0, ..., 1]
    d_phase_gt = h[0, ..., 2]

    h_freq = reconstruction[0, ..., 0]
    h_mag = reconstruction[0, ..., 1]
    d_phase = reconstruction[0, ..., 2]

    print(h_freq.shape, h_freq_gt.shape)

    plt.subplot(2, 1, 1)
    plt.imshow(tf.transpose(h_freq_gt), cmap="binary")
    plt.subplot(2, 1, 2)
    plt.imshow(tf.transpose(h_freq), cmap="binary")
    plt.show()
