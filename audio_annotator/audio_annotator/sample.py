import os

from flask import Blueprint, render_template, flash, redirect, url_for, request
from audio_annotator import db, auth, forms


bp = Blueprint('sample', __name__, url_prefix='/sample')


@bp.route('/<int:sample_id>', methods=['GET', 'POST'])
@auth.login_required
def show_sample(sample_id):
    database = db.get_db()
    if request.method == 'POST':
        database.execute(
            'UPDATE sample SET quality_1 = ?, quality_2 = ? WHERE id = ?',
            (request.form['0'], request.form['1'], sample_id)
        )
        database.commit()
        flash('Annotation saved')

    sample = database.execute(
        'SELECT * FROM sample WHERE id = ?',
        (sample_id, )
    ).fetchone()
    if sample is None:
        flash('No such file found')
        return render_template('index.html')
    else:
        file_name = sample['file_name']
        image_name = f'{os.path.splitext(file_name)[0]}.png'
        items = []
        for i, item in enumerate(forms.qualities_form):
            new_item = item.copy()
            current_value = sample[f'quality_{i + 1}']
            current_index = item['options'].index(current_value)
            new_item['options'][0], new_item['options'][current_index] = current_value, new_item['options'][0]
            items.append(new_item)
        return render_template(
            'sample/show.html',
            file_name=file_name,
            image_name=image_name,
            items=items,
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
