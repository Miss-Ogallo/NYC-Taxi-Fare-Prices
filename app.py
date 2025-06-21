from flask import Flask, request, render_template
import pickle
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load your trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract form data
        pickup_datetime = request.form["pickup_datetime"]
        pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%dT%H:%M")
        pickup_year = pickup_datetime.year
        pickup_month = pickup_datetime.month
        pickup_day = pickup_datetime.day
        pickup_hour = pickup_datetime.hour
        pickup_minute = pickup_datetime.minute

        pickup_latitude = float(request.form["pickup_latitude"])
        pickup_longitude = float(request.form["pickup_longitude"])
        dropoff_latitude = float(request.form["dropoff_latitude"])
        dropoff_longitude = float(request.form["dropoff_longitude"])
        passenger_count = int(request.form["passenger_count"])

        # Prepare features (adjust to your model's input shape/order)
        features = np.array([[
            pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude,
            passenger_count,
            pickup_year, pickup_month, pickup_day, pickup_hour, pickup_minute
        ]])

        # Predict
        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return f"Error: {e}"