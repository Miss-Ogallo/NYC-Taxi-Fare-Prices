import os
import pickle
from flask import Flask, request, jsonify
import numpy as np
import xgboost as xgb  # Ensure this is in requirements.txt

app = Flask(__name__)

# --- Constants ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
FEATURES = [
    'passenger_count', 
    'trip_distance', 
    'pickup_hour',
    'pickup_dayofweek',
    'PULocationID',
    'DOLocationID'
]

# --- Load Model ---
def load_model():
    """Load XGBoost model with error handling"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        raise

model = load_model()

# --- Prediction Logic ---
def validate_input(input_data):
    """Check for required features and types"""
    missing = [f for f in FEATURES if f not in input_data]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    # Convert to float where needed
    return {
        'passenger_count': float(input_data['passenger_count']),
        'trip_distance': float(input_data['trip_distance']),
        'pickup_hour': float(input_data['pickup_hour']),
        'pickup_dayofweek': float(input_data['pickup_dayofweek']),
        'PULocationID': float(input_data['PULocationID']),
        'DOLocationID': float(input_data['DOLocationID'])
    }

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get and validate input
        input_data = request.get_json()
        validated_data = validate_input(input_data)
        
        # 2. Prepare feature array in exact training order
        features = np.array([[validated_data[f] for f in FEATURES]])
        
        # 3. Predict (XGBoost expects 2D array)
        prediction = model.predict(features)[0]
        
        # 4. Convert dollars to cents (avoid float currency)
        return jsonify({
            'fare_cents': int(prediction * 100),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400

# --- Health Check ---
@app.route('/')
def health_check():
    return "Taxi Fare Prediction Service - Healthy ✅"

# --- Run Server ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)