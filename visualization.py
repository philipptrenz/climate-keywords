from flask import Flask, render_template, request, jsonify, make_response

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/filter')
def preset(preset_id):
    return make_response('', 200)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080)
