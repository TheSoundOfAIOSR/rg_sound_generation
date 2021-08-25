import tensorflow as tf
import numpy as np
import os
from tcae.localconfig import LocalConfig
from tcae.dataset import get_dataset
from sound_generator import SoundGenerator
import pickle


def compute_encoding(note_index, velocity_index, instrument_index, conf):
    c0 = conf.num_pitches
    c1 = c0 * conf.num_velocities
    index = note_index + c0 * velocity_index + c1 * instrument_index
    return index


def main():
    sg = SoundGenerator()
    model = sg.model
    conf = sg.conf

    base_path = os.getcwd()
    conf.dataset_dir = os.path.join(base_path, "complete_dataset")
    conf.checkpoints_dir = os.path.join(base_path, "checkpoints")
    conf.batch_size = 1
    conf.data_handler.remap_measures = False
    conf.use_one_hot_conditioning = False
    train_dataset, valid_dataset, test_dataset = get_dataset(conf)

    dataset = train_dataset.concatenate(valid_dataset).concatenate(test_dataset)
    # dataset = test_dataset

    elements = conf.num_pitches * conf.num_velocities * conf.num_instruments
    decoder_inputs = [{"z": None, "measures": None}] * elements

    iterator = iter(dataset)
    for step, batch in enumerate(iterator):
        x, y = batch

        z = model.encoder(x)
        measures = x["measures"]

        note_index = int(tf.math.round(x["note_number"] * conf.num_pitches))
        velocity_index = int(tf.math.round(x["velocity"] * conf.num_velocities))
        instrument_index = int(tf.math.round(x["instrument_id"] * conf.num_instruments))

        index = compute_encoding(
            note_index,
            velocity_index,
            instrument_index,
            conf)

        if decoder_inputs[index]["z"] is None:
            decoder_inputs[index] = {"z": z, "measures": measures}
        else:
            print("wft")

    with open('decoder_inputs.pickle', 'wb') as h:
        pickle.dump(decoder_inputs, h)


if __name__ == "__main__":
    main()
