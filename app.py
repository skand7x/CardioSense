#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import sys

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from predict import predict_single, explain_prediction

app = Flask(__name__)

# Global variables to hold loaded models
rf_model = None
svm_model = None
scaler = None

def load_models():
    """Load the models if they're not already loaded"""
    global rf_model, svm_model, scaler
    
    if rf_model is None or svm_model is None or scaler is None:
        try:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
            svm_model = joblib.load(os.path.join(models_dir, 'svm_model.pkl'))
            scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
            return True
        except FileNotFoundError:
            return False
    return True

@app.route('/')
def index():
    """Render the main page"""
    models_loaded = load_models()
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    if not load_models():
        return jsonify({
            'error': 'Models not found. Please train the models first.'
        }), 500
    
    try:
        # Get patient data from the form
        patient_data = {
            'age': float(request.form.get('age')),
            'sex': int(request.form.get('sex')),
            'cp': int(request.form.get('cp')),
            'trestbps': float(request.form.get('trestbps')),
            'chol': float(request.form.get('chol')),
            'fbs': int(request.form.get('fbs')),
            'restecg': int(request.form.get('restecg')),
            'thalach': float(request.form.get('thalach')),
            'exang': int(request.form.get('exang')),
            'oldpeak': float(request.form.get('oldpeak')),
            'slope': int(request.form.get('slope')),
            'ca': int(request.form.get('ca')),
            'thal': int(request.form.get('thal'))
        }
        
        # Make prediction
        prediction = predict_single(patient_data, (rf_model, svm_model), scaler)
        explanation = explain_prediction(prediction)
        
        # Return results
        return jsonify({
            'prediction': prediction,
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({
            'error': f'Error during prediction: {str(e)}'
        }), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if not load_models():
        return jsonify({
            'error': 'Models not found. Please train the models first.'
        }), 500
    
    try:
        # Get JSON data
        data = request.get_json()
        
        # Make prediction
        prediction = predict_single(data, (rf_model, svm_model), scaler)
        explanation = explain_prediction(prediction)
        
        # Return results
        return jsonify({
            'prediction': prediction,
            'explanation': explanation
        })
    except Exception as e:
        return jsonify({
            'error': f'Error during prediction: {str(e)}'
        }), 400

@app.route('/feature-importance')
def feature_importance():
    """Show feature importance visualization"""
    return render_template('feature_importance.html')

if __name__ == '__main__':
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=5000) 