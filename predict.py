import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tcvae import predict, dataset, localconfig
from matplotlib import pyplot as plt

conf = localconfig.LocalConfig()
conf.load_config_from_file("checkpoints/Default_cnn_decoder_heuristics.json")
conf.dataset_dir = "complete_dataset"
conf.checkpoints_dir = "checkpoints"
conf.batch_size = 1

_, _, test = dataset.get_dataset(conf)
test_iter = iter(test)

decoder = predict.load_model(conf, "checkpoints/14_cnn_decoder_heuristics_0.199.h5")

batch = next(test_iter)

(f0_shifts_true, f0_shifts_pred, mag_env_true, mag_env_pred,
     h_freq_shifts_true, h_freq_shifts_pred, h_mag_dist_true,
     h_mag_dist_pred, mask), note_number_orig = predict.get_intermediate_values(decoder, batch, conf)

plt.figure()
plt.title("F0 Shifts True")
plt.plot(f0_shifts_true[0])

plt.figure()
plt.title("F0 Shifts Pred")
plt.plot(f0_shifts_pred[0])

plt.figure()
plt.title("Mag Env True")
plt.plot(mag_env_true[0])

plt.figure()
plt.title("Mag Env Pred")
plt.plot(mag_env_pred[0])

plt.figure()
plt.title("H Mag Dist True")
plt.plot(h_mag_dist_true[0])

plt.figure()
plt.title("H Mag Dist Pred")
plt.plot(h_mag_dist_pred[0])

# plt.figure()
# plt.plot(h_freq_shifts_true[0])
#
# plt.figure()
# plt.plot(h_freq_shifts_pred[0])

h_freq_true, h_mag_true, h_freq_pred, h_mag_pred = \
    predict.get_freq_and_mag_batch(decoder, batch, conf)

plt.figure()
plt.title("H Mag True")
plt.plot(h_mag_true[0])

plt.figure()
plt.title("H Mag Pred")
plt.plot(h_mag_pred[0])

plt.show()

