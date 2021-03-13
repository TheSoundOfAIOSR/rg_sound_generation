import sqlite3
import os
import click

from flask import current_app, g, Flask
from flask.cli import with_appcontext
from tqdm import tqdm


# Note: g is a special object that is unique for each request
# it can be used to access data that might be accessed by
# multiple functions in the same request
def get_db():
    if 'db' not in g:
        # current_app is another special object that points to the
        # current flask app instance handling the request
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        # sqlite3.Row tells the connection to return rows that behave like dicts
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(*_):
    db = g.pop('db', None)

    if db is not None:
        db.close()


def init_db():
    db = get_db()

    with current_app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf-8'))


@click.command('init-db')
@with_appcontext
def init_db_command():
    init_db()
    click.echo('Initialized the db')


@click.command('build-db')
@click.option('--audio_dir')
@with_appcontext
def build_db_command(audio_dir):
    assert os.path.isdir(audio_dir), f'{audio_dir} must be a directory'
    files = [x for x in os.listdir(audio_dir) if x.lower().endswith('.wav')]
    init_db()
    click.echo('Database initialized')
    db = get_db()
    for file_name in tqdm(files):
        file_path = os.path.join(audio_dir, file_name)
        db.execute(
            'INSERT INTO sample (file_name, file_path, quality_1, quality_2)'
            'VALUES (?, ?, ?, ?)', (file_name, file_path, 'Unsure', 'Unsure')
        )
    db.commit()
    click.echo('Database built')


def init_app(app: Flask):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    app.cli.add_command(build_db_command)
