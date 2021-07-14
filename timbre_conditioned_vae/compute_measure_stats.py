import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tqdm import tqdm
from tcvae.localconfig import LocalConfig
from tcvae.train import get_inputs, get_all_measures
from tcvae.dataset import get_dataset


if __name__ == "__main__":
    conf = LocalConfig()
    conf.batch_size = 1

    print("Loading dataset")

    train, valid, test = get_dataset(conf)
    dataset = train.concatenate(valid)
    dataset = dataset.concatenate(test)

    print("Dataset loaded")

    f = open("heuristic_measures_stats.csv", "w")

    print("Starting computing measures")

    for batch in tqdm(iter(dataset)):
        inputs = get_inputs(batch)
        sample_name = batch["sample_name"][0][0].numpy().decode()
        measures = list(get_all_measures(batch, conf)[0].numpy())
        measures_str = ",".join([str(m) for m in measures])
        out_string = f"{sample_name},{measures_str}\n"
        f.write(out_string)

    f.close()

    print("Finished")
