from flask import Flask, render_template, request
import pickle
import numpy as np
from datetime import datetime
from math import radians, cos, sin, asin, sqrt
from dateutil import parser

app = Flask(__name__)

# Load the trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Haversine function to compute distance between two points
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Landmark coordinates
landmarks = {
    "jfk": (-73.7781, 40.6413),
    "lga": (-73.8740, 40.7769),
    "ewr": (-74.1745, 40.6895),
    "met": (-73.9626, 40.7794),  # Met Museum
    "wtc": (-74.0134, 40.7118),  # World Trade Center
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Get values from form
        pickup_datetime = parser.parse(request.form["pickup_datetime"])
        pickup_lat = float(request.form["pickup_latitude"])
        pickup_lon = float(request.form["pickup_longitude"])
        dropoff_lat = float(request.form["dropoff_latitude"])
        dropoff_lon = float(request.form["dropoff_longitude"])
        passenger_count = int(request.form["passenger_count"])

        # 2. Extract datetime features
        year = pickup_datetime.year
        month = pickup_datetime.month
        day = pickup_datetime.day
        weekday = pickup_datetime.weekday()  # 0 = Monday
        hour = pickup_datetime.hour

        # 3. Calculate distances
        trip_distance = haversine(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
        jfk_distance = haversine(dropoff_lon, dropoff_lat, *landmarks["jfk"])
        lga_distance = haversine(dropoff_lon, dropoff_lat, *landmarks["lga"])
        ewr_distance = haversine(dropoff_lon, dropoff_lat, *landmarks["ewr"])
        met_distance = haversine(dropoff_lon, dropoff_lat, *landmarks["met"])
        wtc_distance = haversine(dropoff_lon, dropoff_lat, *landmarks["wtc"])

        # 4. Create feature vector (match training order)
        features = np.array([[
            pickup_lon, pickup_lat,
            dropoff_lon, dropoff_lat,
            passenger_count,
            year, month, day, weekday, hour,
            trip_distance,
            jfk_distance, lga_distance, ewr_distance,
            met_distance, wtc_distance
        ]])

        # 5. Predict
        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)

        return render_template("home.html", prediction=prediction)

    except Exception as e:
        return f"Prediction error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
