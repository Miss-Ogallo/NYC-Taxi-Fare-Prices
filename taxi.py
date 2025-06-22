from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from datetime import datetime
import traceback

app = Flask(__name__)

# Load your trained model
try:
    model = pickle.load(open('xgb_model.joblib', 'rb'))
except:
    print("Error loading model")
    model = None

@app.route('/')
def home():
    return render_template('home.html')  # Assuming your HTML is saved as home.html

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all form data
        pickup_longitude = float(request.form['pickup_longitude'])
        pickup_latitude = float(request.form['pickup_latitude'])
        dropoff_longitude = float(request.form['dropoff_longitude'])
        dropoff_latitude = float(request.form['dropoff_latitude'])
        passenger_count = int(request.form['passenger_count'])
        
        # Date/time features
        pickup_datetime_year = int(request.form['pickup_datetime_year'])
        pickup_datetime_month = int(request.form['pickup_datetime_month'])
        pickup_datetime_day = int(request.form['pickup_datetime_day'])
        pickup_datetime_weekday = int(request.form['pickup_datetime_weekday'])
        pickup_datetime_hour = int(request.form['pickup_datetime_hour'])
        
        # Distance features
        trip_distance = float(request.form['trip_distance'])
        jfk_drop_distance = float(request.form['jfk_drop_distance'])
        lga_drop_distance = float(request.form['lga_drop_distance'])
        ewr_drop_distance = float(request.form['ewr_drop_distance'])
        met_drop_distance = float(request.form['met_drop_distance'])
        wtc_drop_distance = float(request.form['wtc_drop_distance'])
        
        # Create feature array in the same order as your model expects
        features = [
            pickup_longitude,
            pickup_latitude,
            dropoff_longitude,
            dropoff_latitude,
            passenger_count,
            pickup_datetime_year,
            pickup_datetime_month,
            pickup_datetime_day,
            pickup_datetime_weekday,
            pickup_datetime_hour,
            trip_distance,
            jfk_drop_distance,
            lga_drop_distance,
            ewr_drop_distance,
            met_drop_distance,
            wtc_drop_distance
        ]
        
        # Convert to numpy array and reshape for prediction
        final_features = np.array(features).reshape(1, -1)
        
        # Make prediction
        if model:
            prediction = model.predict(final_features)
            output = round(float(prediction[0]), 2)
            return render_template('home.html', prediction=output)
        else:
            return render_template('home.html', error="Model not loaded")
            
    except Exception as e:
        print(traceback.format_exc())
        return render_template('home.html', error=f"Error processing request: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)