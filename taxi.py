from flask import Flask, request, render_template
import numpy as np
from joblib import load

app = Flask(__name__)

# Load the trained XGBoost model
model = load("xgb_model.joblib")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form and convert to float
        data = [float(x) for x in request.form.values()]
        input_array = np.array(data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_array)[0]
        prediction = round(prediction, 2)

        return render_template('home.html', prediction_text=f"The predicted fare is ${prediction}")
    
    except Exception as e:
        # Show error on page if something goes wrong
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
