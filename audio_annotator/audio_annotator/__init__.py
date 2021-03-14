import os

from dotenv import load_dotenv
from flask import Flask, render_template


def create_app():
    from audio_annotator import db, auth, sample

    load_dotenv()
    app = Flask(__name__, instance_relative_config=True)
    # The configuration files are located in the instance folder
    # which is created outside the audio_annotator package
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('FLASK_SECRET_KEY') or 'xyz',
        DATABASE=os.path.join(app.instance_path, 'db.sqlite')
    )

    try:
        os.makedirs(app.instance_path)
    except OSError:
        print('instance folder already exists')

    @app.route('/')
    def index():
        return render_template('index.html')
    # setup the database
    db.init_app(app)

    # setup the blueprints
    app.register_blueprint(auth.bp)
    app.register_blueprint(sample.bp)

    return app
