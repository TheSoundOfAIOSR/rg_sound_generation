import os

from flask import Blueprint, render_template, flash, redirect, url_for, request
from audio_annotator import db, auth
from audio_annotator.qualities import qualities


bp = Blueprint('sample', __name__, url_prefix='/sample')


@bp.route('/stats', methods=['GET'])
@auth.login_required
def show_stats():
    database = db.get_db()
    query = 'SELECT'
    for q in qualities:
        query += f' SUM(q_{q}),'
    query = query[:-1] + ' FROM sample'
    samples = database.execute(
        query
    ).fetchone()
    counts = dict((qualities[i], s) for i, s in enumerate(samples))
    return render_template('stats.html', counts=counts)


@bp.route('/<int:sample_id>', methods=['GET', 'POST'])
@auth.login_required
def show_sample(sample_id):
    database = db.get_db()
    if request.method == 'POST':
        s = database.execute(
            'SELECT * FROM sample WHERE id = ?',
            (sample_id,)
        ).fetchone()
        if s is None:
            flash('No such file found')
            return render_template('index.html')
        form_data = request.form
        values = [f'q_{q}' in form_data.keys() for q in qualities]
        current_values = [s[f'q_{q}'] for q in qualities]
        updated_values = [v + current_values[i] for i, v in enumerate(values)]
        query = ' = ?, '.join([f'q_{q}' for q in qualities])
        query = 'UPDATE sample SET ' + query + ' = ? WHERE id = ?'
        database.execute(
            query,
            updated_values + [sample_id]
        )
        database.commit()
        flash('Annotation saved')
        return redirect(url_for('sample.next_sample'))
    s = database.execute(
        'SELECT * FROM sample WHERE id = ?',
        (sample_id, )
    ).fetchone()
    if s is None:
        flash('No such file found')
        return render_template('index.html')
    else:
        file_name = s['file_name']
        image_name = f'{os.path.splitext(file_name)[0]}.png'
        return render_template(
            'sample/show.html',
            file_name=file_name,
            image_name=image_name,
            qualities=qualities,
            sample_id=sample_id
        )


@bp.route('/next_sample', methods=['GET'])
def next_sample():
    database = db.get_db()
    sample = database.execute(
        'SELECT * FROM sample ORDER BY RANDOM() LIMIT 1;'
    ).fetchone()
    flash('Loading next sample..')
    return redirect(url_for('sample.show_sample', sample_id=sample['id']))
