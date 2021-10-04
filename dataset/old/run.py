import warnings
import prepare


sets = ["train", "test", "valid"]

dataset_dir = "D:\soundofai\\nsynth-guitar-subset"
checkpoints_dir = "D:\soundofai\\ddsp_trained_30k"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for s in sets:
        # prepare.prepare_partial_tfrecord(dataset_dir=dataset_dir, split=s)
        prepare.prepare_complete_tfrecord(dataset_dir=dataset_dir,
                                          checkpoints_dir=checkpoints_dir, split=s)
