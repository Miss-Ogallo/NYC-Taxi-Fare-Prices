import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the full pipeline (preprocessor + model)
model_pipeline = joblib.load('taxi_pipeline.joblib')

def prepare_features(input_data):
    """Prepare features from form input matching the model's expectations"""

    # Create datetime object
    pickup_datetime = datetime(
        year=int(input_data['pickup_datetime_year']),
        month=int(input_data['pickup_datetime_month']),
        day=int(input_data['pickup_datetime_day']),
        hour=int(input_data['pickup_datetime_hour'])
    )

    # Derive time of day
    hour = pickup_datetime.hour
    time_of_day = pd.cut(
        [hour],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening'],
        right=False
    )[0]

    # Create feature dictionary
    features = {
        'pickup_longitude': float(input_data['pickup_longitude']),
        'pickup_latitude': float(input_data['pickup_latitude']),
        'dropoff_longitude': float(input_data['dropoff_longitude']),
        'dropoff_latitude': float(input_data['dropoff_latitude']),
        'passenger_count': int(input_data['passenger_count']),
        'pickup_datetime_year': pickup_datetime.year,
        'pickup_datetime_month': pickup_datetime.month,
        'pickup_datetime_day': pickup_datetime.day,
        'pickup_datetime_weekday': int(input_data['pickup_datetime_weekday']),
        'pickup_datetime_hour': hour,
        'trip_distance': float(input_data['trip_distance']),
        'jfk_drop_distance': float(input_data.get('jfk_drop_distance', 0)),
        'lga_drop_distance': float(input_data.get('lga_drop_distance', 0)),
        'ewr_drop_distance': float(input_data.get('ewr_drop_distance', 0)),
        'met_drop_distance': float(input_data.get('met_drop_distance', 0)),
        'wtc_drop_distance': float(input_data.get('wtc_drop_distance', 0)),
        'time_of_day': time_of_day  # This will be one-hot encoded inside the pipeline
    }

    return pd.DataFrame([features])

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        features_df = prepare_features(data)

        # Predict using the pipeline
        prediction = model_pipeline.predict(features_df)

        fare = round(float(prediction[0]), 2)
        return render_template('home.html', prediction_text=f"Predicted Fare: ${fare}")

    except Exception as e:
        return render_template('home.html', error=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
