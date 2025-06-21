from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
import os
import pickle

app = Flask(__name__)

# === Model Loading Section ===

model_path = "xgb_model.json"

# If .json model doesn't exist yet, convert from .pkl
if not os.path.exists(model_path):
    print("🔄 Converting model from .pkl to .json...")
    with open("xgb_model.pkl", "rb") as f:
        model_from_pickle = pickle.load(f)
        model_from_pickle.save_model(model_path)
        print("✅ Saved model as xgb_model.json")

# Load the JSON model (permanent format)
model = xgb.XGBRegressor()
model.load_model(model_path)

# === Validation and Feature Prep ===

def validate_inputs(data):
    errors = []
    if not (-74.05 <= data['pickup_longitude'] <= -73.75):
        errors.append("Invalid pickup longitude")
    if not (40.63 <= data['pickup_latitude'] <= 40.85):
        errors.append("Invalid pickup latitude")
    if data['passenger_count'] < 1:
        errors.append("Passenger count must be at least 1")
    return errors

def prepare_features(form_data):
    return np.array([[ 
        form_data['pickup_longitude'],
        form_data['pickup_latitude'],
        form_data['dropoff_longitude'],
        form_data['dropoff_latitude'],
        form_data['passenger_count'],
        form_data['pickup_datetime_year'],
        form_data['pickup_datetime_month'],
        form_data['pickup_datetime_day'],
        form_data['pickup_datetime_weekday'],
        form_data['pickup_datetime_hour'],
        form_data['trip_distance'],
        form_data['jfk_drop_distance'],
        form_data['lga_drop_distance'],
        form_data['ewr_drop_distance'],
        form_data['met_drop_distance'],
        form_data['wtc_drop_distance']
    ]])

@app.route("/")
def home():
    return render_template("home.html", prediction=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        form_data = {
            'pickup_longitude': float(request.form["pickup_longitude"]),
            'pickup_latitude': float(request.form["pickup_latitude"]),
            'dropoff_longitude': float(request.form["dropoff_longitude"]),
            'dropoff_latitude': float(request.form["dropoff_latitude"]),
            'passenger_count': int(request.form["passenger_count"]),
            'pickup_datetime_year': int(request.form["pickup_datetime_year"]),
            'pickup_datetime_month': int(request.form["pickup_datetime_month"]),
            'pickup_datetime_day': int(request.form["pickup_datetime_day"]),
            'pickup_datetime_weekday': int(request.form["pickup_datetime_weekday"]),
            'pickup_datetime_hour': int(request.form["pickup_datetime_hour"]),
            'trip_distance': float(request.form["trip_distance"]),
            'jfk_drop_distance': float(request.form["jfk_drop_distance"]),
            'lga_drop_distance': float(request.form["lga_drop_distance"]),
            'ewr_drop_distance': float(request.form["ewr_drop_distance"]),
            'met_drop_distance': float(request.form["met_drop_distance"]),
            'wtc_drop_distance': float(request.form["wtc_drop_distance"]),
        }

        errors = validate_inputs(form_data)
        if errors:
            return render_template("home.html", error=", ".join(errors), prediction=None)

        features = prepare_features(form_data)
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)

        return render_template("home.html", prediction=prediction, error=None)

    except ValueError:
        return render_template("home.html", error="Invalid input values", prediction=None)
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template("home.html", error="An error occurred during prediction", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
# Run the Flask app
# Note: In production, set debug=False and use a proper WSGI server