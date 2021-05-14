import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import data_loader
import config

plt.style.use("dark_background")

conf = config.get_config()

for f in conf["all_features"]:
    logger.info(f"Plotting mean and std for {f}")
    conf.update({"features": [f]})
    examples = data_loader.data_loader(conf)

    multiple_annots = {}

    for _, value in examples.items():
        file_name = value["audio_file_name"]
        feature_val = value[f]

        if file_name not in multiple_annots:
            multiple_annots[file_name] = [feature_val]
        else:
            multiple_annots[file_name].append(feature_val)

    only_multiple = {}

    for key, value in multiple_annots.items():
        if len(value) > 1:
            only_multiple[key] = value

    stats = {
        "mean": [],
        "std": [],
        "count": [],
        "count_class": []
    }

    for _, value in only_multiple.items():
        stats["mean"].append(np.mean(value))
        stats["std"].append(np.std(value))
        stats["count"].append(len(value))
        stats["count_class"].append(int(len(value) >= 4))

    logger.info("Creating stats")
    df = pd.DataFrame(stats)

    logger.info("Saving plots")

    avg_std = df.loc[:, "std"].mean()

    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(df)), df.loc[:, "std"],
                c=df.loc[:, "count"])
    plt.plot(range(len(df)), [avg_std] * len(df), "--")
    plt.ylim([0., 100.])
    plt.colorbar()
    plt.title(f"standard deviation for {f} with more than 1 annotations")
    plt.savefig(f"standard_deviation_{f}.png")

    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(df)), df.loc[:, "std"],
                c=df.loc[:, "count_class"])
    plt.plot(range(len(df)), [avg_std] * len(df), "--")
    plt.ylim([0., 100.])
    plt.colorbar()
    plt.title(f"standard deviation for {f}, high count (4, 5) vs low (2, 3)")
    plt.savefig(f"standard_deviation_45_{f}.png")

    logger.info(f"{f} done")
