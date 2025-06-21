from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input data from form
        pickup_longitude = float(request.form["pickup_longitude"])
        pickup_latitude = float(request.form["pickup_latitude"])
        dropoff_longitude = float(request.form["dropoff_longitude"])
        dropoff_latitude = float(request.form["dropoff_latitude"])
        passenger_count = int(request.form["passenger_count"])
        pickup_datetime_year = int(request.form["pickup_datetime_year"])
        pickup_datetime_month = int(request.form["pickup_datetime_month"])
        pickup_datetime_day = int(request.form["pickup_datetime_day"])
        pickup_datetime_weekday = int(request.form["pickup_datetime_weekday"])
        pickup_datetime_hour = int(request.form["pickup_datetime_hour"])
        trip_distance = float(request.form["trip_distance"])
        jfk_drop_distance = float(request.form["jfk_drop_distance"])
        lga_drop_distance = float(request.form["lga_drop_distance"])
        ewr_drop_distance = float(request.form["ewr_drop_distance"])
        met_drop_distance = float(request.form["met_drop_distance"])
        wtc_drop_distance = float(request.form["wtc_drop_distance"])

        # Put features in correct order
        features = np.array([[
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
        ]])

        # Make prediction
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)

        return render_template("home.html", prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)

