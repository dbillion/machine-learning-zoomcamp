# ML Zoomcamp Midterm Project

## Project Description
Built a lead scoring classification model using logistic regression with DictVectorizer for feature encoding.

## Features
- Data preprocessing with missing value handling
- Feature engineering with one-hot encoding
- Model training with logistic regression
- Probability prediction for lead conversion

## Model Performance
- AUC: 0.79
- Accuracy: 74%
- F1 Score: 0.47

## Files
- `pipeline_v1.bin`: Trained model pipeline
- `main.py`: Prediction script
- `Dockerfile`: Container configuration

## Usage
```python
import pickle
with open('pipeline_v1.bin', 'rb') as f:
    model = pickle.load(f)
prediction = model.predict_proba([client])[0, 1]
```

## Repository
https://github.com/dbillion/machine-learning-zoomcamp

---
**Status**: Complete ✅
