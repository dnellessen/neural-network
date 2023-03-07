from flask import Flask, render_template, jsonify, request
from processing import *

app = Flask(__name__)


@app.get('/')
def home():
    return render_template('index.html')


@app.post('/image64')
def button_pressed():
    img_data_b64 = request.json['imageDataBase64']
    pred, acti = run_through_network(img_data_b64)
    acti = acti_as_sorted_perc(acti)
    res = {
        'prediction': str(pred),
        'activations': acti
    }
    return jsonify(res)


if __name__ == '__main__':
    print(nn)
    app.run("0.0.0.0", port=8080, debug=True)
