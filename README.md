# Heart Disease Detection

A machine learning application that predicts heart disease risk based on patient symptoms and diagnostics using the Cleveland Heart Disease Dataset from UCI.

## Overview

This application uses two powerful machine learning algorithms:
- **Random Forest**: An ensemble learning method that operates by constructing multiple decision trees
- **Support Vector Machine (SVM)**: A supervised learning model that analyzes data for classification

Both models are trained on the Cleveland Heart Disease Dataset from the UCI Machine Learning Repository.

## Features

- Data preprocessing and cleaning
- Model training with hyperparameter tuning using GridSearchCV
- Feature importance visualization
- Model evaluation with confusion matrices
- Web interface for making predictions
- RESTful API for integration with other systems
- Ensemble approach combining predictions from both models

## Usage

### Web Interface

The web application provides an easy-to-use interface where you can:
1. Enter patient data
2. Get predictions from both models
3. View an ensemble prediction with risk assessment
4. Explore feature importance and model performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- Cleveland Heart Disease Dataset: UCI Machine Learning Repository
- Scikit-learn for machine learning implementation
- Flask for web application framework 
