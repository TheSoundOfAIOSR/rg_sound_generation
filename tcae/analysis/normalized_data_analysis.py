import numpy as np
import os
from tcae.localconfig import LocalConfig
from tcae.dataset import get_dataset


def main():
    conf = LocalConfig()
    base_path = os.getcwd()
    base_path = os.path.dirname(base_path)
    base_path = os.path.dirname(base_path)
    conf.dataset_dir = os.path.join(base_path, "complete_dataset")
    conf.checkpoints_dir = os.path.join(base_path, "checkpoints")
    train_dataset, valid_dataset, test_dataset = get_dataset(conf)

    dataset = train_dataset.concatenate(valid_dataset).concatenate(test_dataset)

    f0_shifts_max = 0.0
    mag_env_max = 0.0
    h_freq_shifts_max = 0.0
    h_mag_dist_max = 0.0
    h_phase_diff_max = 0.0
    h_mag_max = 0.0

    f0_shifts_means = []
    mag_env_means = []
    h_freq_shifts_means = []
    h_mag_dist_means = []
    h_phase_diff_means = []
    h_mag_means = []

    f0_shifts_variances = []
    mag_env_variances = []
    h_freq_shifts_variances = []
    h_mag_dist_variances = []
    h_phase_diff_variances = []
    h_mag_variances = []

    iterator = iter(dataset)
    for step, batch in enumerate(iterator):
        x, y = batch
        f0_shifts = y["f0_shifts"].numpy()
        mag_env = y["mag_env"].numpy()
        h_freq_shifts = y["h_freq_shifts"].numpy()
        h_mag_dist = y["h_mag_dist"].numpy()
        h_phase_diff = y["h_phase_diff"].numpy()
        h_mag = conf.data_handler.combine_mag(mag_env, h_mag_dist)

        f0_shifts_max = np.maximum(f0_shifts_max, np.amax(np.abs(f0_shifts)))
        mag_env_max = np.maximum(mag_env_max, np.amax(np.abs(mag_env)))
        h_freq_shifts_max = np.maximum(h_freq_shifts_max, np.amax(np.abs(h_freq_shifts)))
        h_mag_dist_max = np.maximum(h_mag_dist_max, np.amax(np.abs(h_mag_dist)))
        h_phase_diff_max = np.maximum(h_phase_diff_max, np.amax(np.abs(h_phase_diff)))
        h_mag_max = np.maximum(h_mag_max, np.amax(np.abs(h_mag)))

        f0_shifts_means.append(np.mean(f0_shifts))
        mag_env_means.append(np.mean(mag_env))
        h_freq_shifts_means.append(np.mean(h_freq_shifts))
        h_mag_dist_means.append(np.mean(h_mag_dist))
        h_phase_diff_means.append(np.mean(h_phase_diff))
        h_mag_means.append(np.mean(h_mag))

    f0_shifts_mean = np.mean(f0_shifts_means)
    mag_env_mean = np.mean(mag_env_means)
    h_freq_shifts_mean = np.mean(h_freq_shifts_means)
    h_mag_dist_mean = np.mean(h_mag_dist_means)
    h_phase_diff_mean = np.mean(h_phase_diff_means)
    h_mag_mean = np.mean(h_mag_means)

    train_iterator = iter(train_dataset)
    for step, batch in enumerate(train_iterator):
        x, y = batch
        f0_shifts = x["f0_shifts"].numpy()
        mag_env = y["mag_env"].numpy()
        h_freq_shifts = y["h_freq_shifts"].numpy()
        h_mag_dist = y["h_mag_dist"].numpy()
        h_phase_diff = y["h_phase_diff"].numpy()
        h_mag = conf.data_handler.combine_mag(mag_env, h_mag_dist)

        f0_shifts_variances.append(np.mean((f0_shifts - f0_shifts_mean) ** 2))
        mag_env_variances.append(np.mean((mag_env - mag_env_mean) ** 2))
        h_freq_shifts_variances.append(np.mean((h_freq_shifts - h_freq_shifts_mean) ** 2))
        h_mag_dist_variances.append(np.mean((h_mag_dist - h_mag_dist_mean) ** 2))
        h_phase_diff_variances.append(np.mean((h_phase_diff - h_phase_diff_mean) ** 2))
        h_mag_variances.append(np.mean((h_mag - h_mag_mean) ** 2))

    f0_shifts_variance = np.mean(f0_shifts_variances)
    mag_env_variance = np.mean(mag_env_variances)
    h_freq_shifts_variance = np.mean(h_freq_shifts_variances)
    h_mag_dist_variance = np.mean(h_mag_dist_variances)
    h_phase_diff_variance = np.mean(h_phase_diff_variances)
    h_mag_variance = np.mean(h_mag_variances)

    f0_shifts_std = np.sqrt(f0_shifts_variance)
    mag_env_std = np.sqrt(mag_env_variance)
    h_freq_shifts_std = np.sqrt(h_freq_shifts_variance)
    h_mag_dist_std = np.sqrt(h_mag_dist_variance)
    h_phase_diff_std = np.sqrt(h_phase_diff_variance)
    h_mag_std = np.sqrt(h_mag_variance)

    print("f0_shifts_max: ", f0_shifts_max)
    print("mag_env_max: ", mag_env_max)
    print("h_freq_shifts_max: ", h_freq_shifts_max)
    print("h_mag_dist_max: ", h_mag_dist_max)
    print("h_phase_diff_max: ", h_phase_diff_max)
    print("h_mag_max: ", h_mag_max)
    print("")
    print("f0_shifts_mean: ", f0_shifts_mean)
    print("mag_env_mean: ", mag_env_mean)
    print("h_freq_shifts_mean: ", h_freq_shifts_mean)
    print("h_mag_dist_mean: ", h_mag_dist_mean)
    print("h_phase_diff_mean: ", h_phase_diff_mean)
    print("h_mag_mean: ", h_mag_mean)
    print("")
    print("f0_shifts_variance: ", f0_shifts_variance)
    print("mag_env_variance: ", mag_env_variance)
    print("h_freq_shifts_variance: ", h_freq_shifts_variance)
    print("h_mag_dist_variance: ", h_mag_dist_variance)
    print("h_phase_diff_variance: ", h_phase_diff_variance)
    print("h_mag_variance: ", h_mag_variance)
    print("")
    print("f0_shifts_std: ", f0_shifts_std)
    print("mag_env_std: ", mag_env_std)
    print("h_freq_shifts_std: ", h_freq_shifts_std)
    print("h_mag_dist_std: ", h_mag_dist_std)
    print("h_phase_diff_std: ", h_phase_diff_std)
    print("h_mag_std: ", h_mag_std)


if __name__ == "__main__":
    main()
