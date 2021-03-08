from flask import Blueprint, render_template, request, make_response
from annotator.db import get_random, add_qualities
from annotator.audio import get_spectrogram


bp = Blueprint('main', __name__)


@bp.route('/', methods=('GET', 'POST'))
def index():
    if request.method == 'POST':
        new_qualities = [request.form.get(f'q{i}') or '' for i in range(1, 7)]
        key = request.cookies.get('sample_key')
        add_qualities(key, new_qualities)
    key, audio_path, qualities = get_random()
    image_path = get_spectrogram(audio_path, key)
    resp = make_response(render_template(
        'index.html',
        key=key,
        audio_path=audio_path,
        qualities=qualities,
        image_path=image_path
    ))
    resp.set_cookie('sample_key', key)
    return resp
