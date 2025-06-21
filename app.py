from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

def validate_inputs(data):
    """Validate form inputs"""
    errors = []
    
    # Validate coordinates (rough NYC bounds)
    if not (-74.05 <= data['pickup_longitude'] <= -73.75):
        errors.append("Invalid pickup longitude")
    if not (40.63 <= data['pickup_latitude'] <= 40.85):
        errors.append("Invalid pickup latitude")
    # Similar validation for dropoff coordinates
    
    if data['passenger_count'] < 1:
        errors.append("Passenger count must be at least 1")
        
    # Add more validations as needed
    
    return errors

def prepare_features(form_data):
    """Prepare features in correct order for model prediction"""
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
    return render_template("home.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get all form data
        form_data = {
            'pickup_longitude': float(request.form["pickup_longitude"]),
            'pickup_latitude': float(request.form["pickup_latitude"]),
            # ... all other fields ...
        }
        
        # Validate inputs
        errors = validate_inputs(form_data)
        if errors:
            return render_template("home.html", 
                                 error=", ".join(errors),
                                 prediction=None)
        
        # Prepare features
        features = prepare_features(form_data)
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)

        return render_template("home.html", 
                             prediction=prediction,
                             error=None)

    except ValueError as e:
        return render_template("home.html", 
                             error="Invalid input values",
                             prediction=None)
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template("home.html", 
                             error="An error occurred during prediction",
                             prediction=None)

if __name__ == "__main__":
    app.run(debug=True)  # Set debug=False in production