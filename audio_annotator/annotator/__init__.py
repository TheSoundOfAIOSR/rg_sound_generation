from flask import Flask
from annotator.main import bp


app = Flask(__name__)
app.register_blueprint(bp)
