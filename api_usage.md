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
