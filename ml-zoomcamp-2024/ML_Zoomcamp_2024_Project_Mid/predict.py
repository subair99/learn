import pickle
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'xgb_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('arrhythmia')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.inplace_predict(X)
    arrhythmia = y_pred >= 0.5

    result = {
        'arrhythmia_probability': float(y_pred),
        'arrhythmia': bool(arrhythmia)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
    