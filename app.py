from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        f1 = float(request.form["f1"])
        f2 = float(request.form["f2"])
        f3 = float(request.form["f3"])
        f4 = float(request.form["f4"])
        features = np.array([[f1, f2, f3, f4]])
        prediction = model.predict(features)[0]
        return render_template("index.html", prediction=prediction)
    except Exception as e:
        return f"Error: {e}"

