import json
from flask import Flask, render_template, request, Response

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/data', methods=["POST"])
def preset():
    data = request.get_json() # array of keywords

    # TODO: filter corpora and return data

    result = {
        "data1": [0, 5, 1, 2, 7, 5, 6, 15, 24, 7, 12, 5, 6, 3, 2, 2, 6, 30, 10, 32, 15, 14, 47, 65, 55],
        "data2": [0, 5, 1, 2, 7, 5, 6, 15, 24, 7, 12, 5, 6, 3, 2, 2, 6, 30, 10, 0, 15, 14, 47, 65, 8],
        "data3": [0, 5, 1, 2, 7, 5, 6, 4, 24, 7, 12, 5, 6, 3, 2, 2, 6, 30, 10, 0, 15, 14, 47, 65, 24]
    }

    return Response(json.dumps(result),  mimetype='application/json')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)
