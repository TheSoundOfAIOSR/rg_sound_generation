from tcvae import predict, dataset, localconfig
from matplotlib import pyplot as plt

conf = localconfig.LocalConfig()
conf.load_config_from_file("C:/new_data/checkpoints/Default_cnn_decoder_heuristics.json")
conf.dataset_dir = "C:/new_data"
conf.checkpoints_dir = "C:/new_data/checkpoints"
conf.batch_size = 1

_, _, test = dataset.get_dataset(conf)
test_iter = iter(test)

decoder = predict.load_model(conf, "C:/new_data/checkpoints/48_cnn_decoder_heuristics_0.03871.h5")

batch = next(test_iter)

(f0_shifts_true, f0_shifts_pred, mag_env_true, mag_env_pred,
     h_freq_shifts_true, h_freq_shifts_pred, h_mag_dist_true,
     h_mag_dist_pred, mask), note_number_orig = predict.dummy(decoder, batch, conf)

plt.figure()
plt.plot(f0_shifts_true[0])

plt.figure()
plt.plot(f0_shifts_pred[0])

plt.figure()
plt.plot(mag_env_true[0])

plt.figure()
plt.plot(mag_env_pred[0])

plt.figure()
plt.plot(h_mag_dist_true[0])

plt.figure()
plt.plot(h_mag_dist_pred[0])

# plt.figure()
# plt.plot(h_freq_shifts_true[0])
#
# plt.figure()
# plt.plot(h_freq_shifts_pred[0])

h_freq_true, h_mag_true, h_freq_pred, h_mag_pred = \
    predict.get_freq_and_mag_batch(decoder, batch, conf)

plt.figure()
plt.plot(h_mag_true[0])

plt.figure()
plt.plot(h_mag_pred[0])

plt.show()

