import sqlite3
import os
import click

from flask import current_app, g, Flask
from flask.cli import with_appcontext
from tqdm import tqdm
from audio_annotator.qualities import qualities


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
@with_appcontext
def build_db_command():
    audio_dir = os.path.join('audio_annotator', 'static')
    files = [x for x in os.listdir(audio_dir) if x.lower().endswith('.wav')]
    init_db()
    click.echo('Database initialized')
    db = get_db()
    for file_name in tqdm(files):
        file_path = os.path.join(audio_dir, file_name)
        query = 'INSERT INTO sample (file_name, file_path, ' + ', '.join([f'q_{q}' for q in qualities])
        query += ') VALUES (?, ?, ' + '?,' * (len(qualities) - 1) + '?)'
        db.execute(
            query, [file_name, file_path, ] + [0] * len(qualities),
        )
    db.commit()
    click.echo('Database built')


@click.command('generate-spectrograms')
def generate_spectrograms():
    from audio_annotator.create_spectrograms import create_spectrograms
    create_spectrograms()


@click.command('load-pre-db')
@with_appcontext
def load_pre_db():
    import pandas as pd

    if not os.path.isfile('pre.csv'):
        click.echo('Could not find pre.csv, no data to pre-load')
        return

    df = pd.read_csv('pre.csv', index_col=0)
    db = get_db()

    click.echo('Loading data from pre.csv')

    for row in tqdm(df.iterrows()):
        query = ' = ?, '.join([f'q_{q}' for q in qualities])
        query = 'UPDATE sample SET ' + query + ' = ?, description = ? WHERE id = ?'

        row_id = row[0]

        values = [df.loc[row_id, f'q_{q}'] for q in qualities]
        description = df.loc[row_id, 'description']
        if str(description).lower() == 'nan':
            description = ''
        else:
            description = str(description)
        values.append(description)
        values.append(row_id)

        db.execute(
            query,
            values
        )


def init_app(app: Flask):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    app.cli.add_command(build_db_command)
    app.cli.add_command(generate_spectrograms)
    app.cli.add_command(load_pre_db)
