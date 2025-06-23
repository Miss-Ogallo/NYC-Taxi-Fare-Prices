from flask import Flask, request, render_template
import numpy as np
from joblib import load

app = Flask(__name__)
model = load("xgb_model.joblib")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form values in exact model training order
        form_keys = [
            "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
            "passenger_count", "pickup_datetime_year", "pickup_datetime_month",
            "pickup_datetime_day", "pickup_datetime_weekday", "pickup_datetime_hour",
            "trip_distance", "jfk_drop_distance", "lga_drop_distance",
            "ewr_drop_distance", "met_drop_distance", "wtc_drop_distance"
        ]

        data = [float(request.form.get(k)) for k in form_keys]
        input_array = np.array(data).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        prediction = round(prediction, 2)

        return render_template('home.html', prediction_text=f"The predicted fare is ${prediction}")

    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

