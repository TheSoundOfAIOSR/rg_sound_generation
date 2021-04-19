import os
import click


@click.command()
@click.option("--source")
@click.option("--target")
def start(source, target):
    files = [x for x in os.listdir(source) if x.lower().endswith(".wav")]

    click.echo(f"Found {len(files)} files")

    for file_name in files:
        *_, properties = file_name.split("_")
        numeric = properties.split(".")[0]
        _, pitch, _ = numeric.split("-")
        pitch = int(pitch)
        delta = 60 - pitch
        if delta != 0 and abs(delta) < 6:
            file_path = os.path.join(source, file_name)
            target_path = os.path.join(target, f"p+{file_name}")
            command = f"rubberband --pitch {delta:0.5f} {file_path} {target_path}"
            click.echo(command)
            os.system(command)


if __name__ == "__main__":
    start()
