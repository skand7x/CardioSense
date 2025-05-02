#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create data directory if it doesn't exist
os.makedirs('../data', exist_ok=True)

# Download the Cleveland Heart Disease dataset if not already downloaded
def download_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        # Replace '?' with NaN and clean the dataset
        df = df.replace('?', np.nan)
        # Convert columns to appropriate data types
        for col in df.columns:
            if col != 'target':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Save to data directory
        df.to_csv('../data/heart_disease_data.csv', index=False)
        print("Dataset downloaded and saved to data/heart_disease_data.csv")
        return df
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def load_dataset():
    try:
        # First try to load the local file
        return pd.read_csv('../data/heart_disease_data.csv')
    except FileNotFoundError:
        # If not found, download it
        return download_dataset()

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()
    
    # Convert the target to binary (0 = no disease, 1 = disease)
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for later use in predictions
    joblib.dump(scaler, '../models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_random_forest(X_train, y_train):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Create the model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_rf = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_rf, '../models/random_forest_model.pkl')
    
    print(f"Random Forest Best Parameters: {grid_search.best_params_}")
    
    return best_rf

def train_svm(X_train, y_train):
    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1]
    }
    
    # Create the model
    svm = SVC(probability=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_svm = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_svm, '../models/svm_model.pkl')
    
    print(f"SVM Best Parameters: {grid_search.best_params_}")
    
    return best_svm

def evaluate_model(model, X_test, y_test, model_name):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'../static/{model_name}_confusion_matrix.png')
    
    # Plot feature importance for Random Forest
    if model_name == 'Random Forest' and hasattr(model, 'feature_importances_'):
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances - Random Forest')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig('../static/feature_importance.png')
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Classification Report:\n{report}")

def main():
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Load and preprocess data
    df = load_dataset()
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train and evaluate Random Forest
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Train and evaluate SVM
    svm_model = train_svm(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, "SVM")
    
    # Save the processed test data for later demonstration
    np.save('../data/X_test.npy', X_test)
    np.save('../data/y_test.npy', y_test)

if __name__ == "__main__":
    main() 