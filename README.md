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

## Requirements

- Python 3.7+
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/heart-disease-detection.git
   cd heart-disease-detection
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Train the models:
   ```
   cd scripts
   python train_model.py
   cd ..
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to: `http://localhost:5000`

## Usage

### Web Interface

The web application provides an easy-to-use interface where you can:
1. Enter patient data
2. Get predictions from both models
3. View an ensemble prediction with risk assessment
4. Explore feature importance and model performance

### API Usage

You can also use the RESTful API to integrate with other systems:

```python
import requests
import json

# Example patient data
patient_data = {
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

# Make prediction request
response = requests.post('http://localhost:5000/api/predict', 
                        json=patient_data,
                        headers={'Content-Type': 'application/json'})

# Parse results
result = response.json()
print(result['explanation'])
print("Random Forest Probability:", result['prediction']['random_forest']['probability'])
print("SVM Probability:", result['prediction']['svm']['probability'])
print("Ensemble Prediction:", "Positive" if result['prediction']['ensemble']['prediction'] == 1 else "Negative")
```

## Data Dictionary

The Cleveland Heart Disease dataset contains the following features:

1. **age**: Age in years
2. **sex**: Sex (1 = male, 0 = female)
3. **cp**: Chest pain type
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Resting blood pressure (in mm Hg)
5. **chol**: Serum cholesterol in mg/dl
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results
   - 0: Normal
   - 1: Having ST-T wave abnormality
   - 2: Showing probable or definite left ventricular hypertrophy
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: The slope of the peak exercise ST segment
    - 0: Upsloping
    - 1: Flat
    - 2: Downsloping
12. **ca**: Number of major vessels (0-3) colored by fluoroscopy
13. **thal**: Thalassemia
    - 0: Normal
    - 1: Fixed defect
    - 2: Reversible defect
14. **target**: Heart disease diagnosis (1 = present, 0 = absent)

## Deployment Options

This application can be deployed in several ways:

### Local Deployment

Run the application locally using the Flask development server (as shown in the installation section).

### Docker Deployment

1. Build the Docker image:
   ```
   docker build -t heart-disease-predictor .
   ```

2. Run the container:
   ```
   docker run -p 5000:5000 heart-disease-predictor
   ```

### Cloud Deployment

The application can be deployed to various cloud platforms:

1. **Heroku**:
   ```
   heroku login
   heroku create heart-disease-predictor
   git push heroku main
   ```

2. **AWS Elastic Beanstalk**:
   - Create an application in the AWS Elastic Beanstalk console
   - Deploy the code using the AWS CLI or the EB CLI

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- Cleveland Heart Disease Dataset: UCI Machine Learning Repository
- Scikit-learn for machine learning implementation
- Flask for web application framework 