import json

from flask import Flask, request, jsonify
from sound_generator import get_prediction


app = Flask(__name__)


@app.route('/sound/', methods=['POST'])
def api():
    data = json.loads(request.data)
    prediction = get_prediction(data)
    if prediction is not None:
        return jsonify({'audio': prediction.tolist()}), 200
    return jsonify({}), 400


if __name__ == '__main__':
    app.run()
