from flask import Flask, request, jsonify, render_template
import numpy as np
from joblib import load

app = Flask(__name__)

# ✅ Load the trained XGBoost model
model = load('xgb_model.joblib')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        input_array = np.array(list(data.values())).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        return jsonify({'prediction': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values and convert to float list
        data = [float(x) for x in request.form.values()]
        input_array = np.array(data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        return render_template('home.html', prediction_text=f'The predicted fare is ${round(prediction, 2)}')
    except Exception as e:
        return render_template('home.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
