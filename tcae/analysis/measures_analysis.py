import tensorflow as tf
import numpy as np
import os
from tcae.localconfig import LocalConfig
from tcae.dataset import get_dataset
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle


def main(export_map=False):
    conf = LocalConfig()
    # base_path = os.getcwd()
    # base_path = os.path.dirname(base_path)
    # base_path = os.path.dirname(base_path)
    # conf.dataset_dir = os.path.join(base_path, "complete_dataset")
    # conf.checkpoints_dir = os.path.join(base_path, "checkpoints")
    conf.batch_size = 1
    conf.data_handler.remap_measures = False
    conf.use_one_hot_conditioning = False
    train_dataset, valid_dataset, _ = get_dataset(conf)

    dataset = train_dataset.concatenate(valid_dataset)
    # dataset = test_dataset

    measure_names = conf.data_handler.measure_names
    measures = dict((k, []) for k in measure_names)

    matrix = np.zeros(shape=(conf.num_pitches, conf.num_velocities))
    measures_mean_matrix = dict((k, matrix.copy()) for k in measure_names)
    measures_count_matrix = dict((k, matrix.copy()) for k in measure_names)

    iterator = iter(dataset)
    for step, batch in enumerate(iterator):
        x, y = batch

        note_number = int(tf.math.round(x["note_number"] * conf.num_pitches))
        velocity = int(tf.math.round(x["velocity"] * conf.num_velocities))

        # y = conf.data_handler.shift_measures_mean(
        #     y, note_number, velocity)

        for k in measure_names:
            value = tf.squeeze(y[k]).numpy()
            value = float(value)
            measures[k].append(value)
            measures_mean_matrix[k][note_number, velocity] += value
            measures_count_matrix[k][note_number, velocity] += 1.0

    for k in measure_names:
        measures_mean_matrix[k] = np.where(
            measures_count_matrix[k] > 0.0,
            measures_mean_matrix[k] / measures_count_matrix[k], 0.0)

        measures_mean_matrix[k] = tf.cast(
            measures_mean_matrix[k], dtype=tf.float32)

    x = np.arange(conf.num_velocities, dtype=np.float32)
    y = np.arange(conf.num_pitches, dtype=np.float32)
    x, y = np.meshgrid(x, y)

    for i, (k, z) in enumerate(measures_mean_matrix.items()):
        plt.figure()
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(x, y, z, label=k)
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d
        ax.legend()

    plt.show()

    with open('measures_mean_matrix.pickle', 'wb') as h:
        pickle.dump(measures_mean_matrix, h)

    if export_map:
        measures_map = dict((k, {}) for k in measure_names)

        plt.figure()
        for i, (k, v) in enumerate(measures.items()):
            y = np.sort(np.asarray(v))
            y_min = np.min(y)
            y_max = np.max(y)
            y = (y - y_min) / (y_max - y_min)
            x = np.linspace(0, 1, np.size(y))

            f = interpolate.interp1d(x, y)
            f_inv = interpolate.interp1d(y, x)

            x_new = np.linspace(0, 1, 10000)

            y_new = f(x_new)
            y_new_inv = f_inv(x_new)

            measures_map[k]["y_min"] = tf.cast(y_min, dtype=tf.float32)
            measures_map[k]["y_max"] = tf.cast(y_max, dtype=tf.float32)
            measures_map[k]["y"] = tf.cast(y_new, dtype=tf.float32)
            measures_map[k]["y_inv"] = tf.cast(y_new_inv, dtype=tf.float32)

            plt.subplot(11, 2, 2*i+1)
            plt.plot(x_new, y_new, label=k)
            plt.legend()
            plt.subplot(11, 2, 2*i+2)
            plt.plot(x_new, y_new_inv, label=k + '_inv')
            plt.legend()

        with open('measures_map.pickle', 'wb') as h:
            pickle.dump(measures_map, h)

        plt.show()


if __name__ == "__main__":
    main()
