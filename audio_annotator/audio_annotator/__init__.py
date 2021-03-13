import os

from flask import Flask, render_template


def create_app(test_config=None):
    from audio_annotator import db, auth, sample

    app = Flask(__name__, instance_relative_config=True)
    # The configuration files are located in the instance folder
    # which is created outside the audio_annotator package
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'db.sqlite')
    )

    # if test_config is None:
    #     app.config.from_pyfile('config.py')
    # else:
    #     app.config.from_mapping(test_config)

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
