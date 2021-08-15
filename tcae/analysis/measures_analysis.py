import tensorflow as tf
import numpy as np
import os
from tcae.localconfig import LocalConfig
from tcae.dataset import get_dataset
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle


def main():
    conf = LocalConfig()
    base_path = os.getcwd()
    base_path = os.path.dirname(base_path)
    base_path = os.path.dirname(base_path)
    conf.dataset_dir = os.path.join(base_path, "complete_dataset")
    conf.checkpoints_dir = os.path.join(base_path, "checkpoints")
    conf.batch_size = 1
    conf.data_handler.remap_measures = False
    train_dataset, valid_dataset, test_dataset = get_dataset(conf)

    dataset = train_dataset.concatenate(valid_dataset).concatenate(test_dataset)
    # dataset = test_dataset

    measures_names = conf.data_handler.measures_names
    measures = dict((k, []) for k in measures_names)

    iterator = iter(dataset)
    for step, batch in enumerate(iterator):
        x, y = batch

        for k, v in measures.items():
            value = tf.squeeze(y[k]).numpy()
            value = float(value)
            measures[k].append(value)

    measures_map = dict((k, {}) for k in measures_names)

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

    with open('measures_map0.pickle', 'wb') as h:
        pickle.dump(measures_map, h)

    plt.show()


if __name__ == "__main__":
    main()
