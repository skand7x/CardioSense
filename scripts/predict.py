#!/usr/bin/env python3
import joblib
import numpy as np
import pandas as pd
import os

def load_models():
    """Load the trained models and scaler"""
    models_dir = '../models'
    
    # Check if models exist
    rf_path = os.path.join(models_dir, 'random_forest_model.pkl')
    svm_path = os.path.join(models_dir, 'svm_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    if not (os.path.exists(rf_path) and os.path.exists(svm_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError("Models not found. Please run train_model.py first.")
    
    # Load models and scaler
    rf_model = joblib.load(rf_path)
    svm_model = joblib.load(svm_path)
    scaler = joblib.load(scaler_path)
    
    return rf_model, svm_model, scaler

def predict_single(features, models=None, scaler=None):
    """
    Make a prediction on a single patient's data
    
    Parameters:
    - features: Dictionary with feature names and values
    - models: Tuple of (rf_model, svm_model) if provided, otherwise loaded from disk
    - scaler: Scaler object if provided, otherwise loaded from disk
    
    Returns:
    - Dictionary with predictions and probabilities from both models
    """
    if models is None or scaler is None:
        rf_model, svm_model, scaler = load_models()
    else:
        rf_model, svm_model = models
    
    # Convert features to DataFrame for consistent processing
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    # Create feature array making sure the order matches what the model expects
    features_array = np.array([features[name] for name in feature_names]).reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(features_array)
    
    # Make predictions
    rf_pred = rf_model.predict(scaled_features)[0]
    rf_prob = rf_model.predict_proba(scaled_features)[0][1]  # Probability of class 1
    
    svm_pred = svm_model.predict(scaled_features)[0]
    svm_prob = svm_model.predict_proba(scaled_features)[0][1]  # Probability of class 1
    
    # Ensemble prediction (majority vote)
    ensemble_pred = 1 if (rf_pred + svm_pred) >= 1 else 0
    ensemble_prob = (rf_prob + svm_prob) / 2
    
    return {
        'random_forest': {
            'prediction': int(rf_pred),
            'probability': float(rf_prob)
        },
        'svm': {
            'prediction': int(svm_pred),
            'probability': float(svm_prob)
        },
        'ensemble': {
            'prediction': int(ensemble_pred),
            'probability': float(ensemble_prob)
        }
    }

def explain_prediction(prediction_result):
    """
    Generate a human-readable explanation of the prediction
    
    Parameters:
    - prediction_result: Dictionary returned by predict_single function
    
    Returns:
    - String with explanation
    """
    rf_pred = prediction_result['random_forest']['prediction']
    rf_prob = prediction_result['random_forest']['probability']
    
    svm_pred = prediction_result['svm']['prediction']
    svm_prob = prediction_result['svm']['probability']
    
    ensemble_pred = prediction_result['ensemble']['prediction']
    ensemble_prob = prediction_result['ensemble']['probability']
    
    result = "Heart Disease Prediction Results:\n\n"
    
    # Random Forest results
    result += f"Random Forest: {'Positive' if rf_pred == 1 else 'Negative'} "
    result += f"(Confidence: {rf_prob:.2%})\n"
    
    # SVM results
    result += f"Support Vector Machine: {'Positive' if svm_pred == 1 else 'Negative'} "
    result += f"(Confidence: {svm_prob:.2%})\n"
    
    # Ensemble results
    result += f"Ensemble Prediction: {'Positive' if ensemble_pred == 1 else 'Negative'} "
    result += f"(Confidence: {ensemble_prob:.2%})\n\n"
    
    # Overall interpretation
    if ensemble_pred == 1:
        if ensemble_prob > 0.8:
            result += "Interpretation: High risk of heart disease. Please consult a healthcare professional immediately."
        elif ensemble_prob > 0.6:
            result += "Interpretation: Moderate risk of heart disease. Consider consulting a healthcare professional."
        else:
            result += "Interpretation: Slight risk of heart disease. Consider following up with a healthcare professional."
    else:
        if ensemble_prob < 0.2:
            result += "Interpretation: Very low risk of heart disease."
        else:
            result += "Interpretation: Low risk of heart disease, but maintaining heart-healthy habits is always recommended."
    
    return result

if __name__ == "__main__":
    # Example usage
    example_patient = {
        'age': 63,
        'sex': 1,  # 1 for male, 0 for female
        'cp': 3,   # chest pain type (0-3)
        'trestbps': 145,  # resting blood pressure
        'chol': 233,      # serum cholesterol in mg/dl
        'fbs': 1,         # fasting blood sugar > 120 mg/dl (1=true, 0=false)
        'restecg': 0,     # resting electrocardiographic results (0-2)
        'thalach': 150,   # maximum heart rate achieved
        'exang': 0,       # exercise induced angina (1=yes, 0=no)
        'oldpeak': 2.3,   # ST depression induced by exercise relative to rest
        'slope': 0,       # the slope of the peak exercise ST segment (0-2)
        'ca': 0,          # number of major vessels (0-3) colored by fluoroscopy
        'thal': 1         # thalassemia (0=normal, 1=fixed defect, 2=reversible defect)
    }
    
    try:
        prediction = predict_single(example_patient)
        explanation = explain_prediction(prediction)
        print(explanation)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the models first by running: python train_model.py") 