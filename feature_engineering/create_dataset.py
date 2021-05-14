import os
import click
from loguru import logger
from tqdm import tqdm
from sample_analysis import decompose

from multiprocessing import Process


@click.command()
@click.option("--dir")
def create_decomposed_files(dir: str):
    logger.info("starting")
    assert os.path.isdir(dir), f"{dir} is not a valid directory"
    files = [x for x in os.listdir(dir) if x.lower().endswith(".wav")]
    logger.info(f"found {len(files)} to process")

    num_processes = 2

    for indices in tqdm(range(0, len(files), num_processes)):
        processes = []
        for i in range(indices, indices + num_processes):
            file_path = os.path.join(dir, files[i])
            processes.append(Process(target=decompose, args=(file_path, False, False)))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

    logger.info("finished")


if __name__ == "__main__":
    create_decomposed_files()
