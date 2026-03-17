"""
ML Zoomcamp Capstone: FastAPI Taxi Fare Prediction API
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Taxi Fare Prediction API", version="1.0.0")

# Load model
with open('taxi_fare_model.pkl', 'rb') as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    passenger_count: int
    trip_distance: float
    pickup_hour: int
    pickup_dayofweek: int
    PULocationID: int
    DOLocationID: int

class PredictionResponse(BaseModel):
    predicted_fare: float
    currency: str = "USD"

@app.get("/")
def root():
    return {"message": "Taxi Fare Prediction API", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        # Prepare features
        features = pd.DataFrame({
            'passenger_count': [request.passenger_count],
            'trip_distance_km': [request.trip_distance * 1.60934],
            'pickup_hour': [request.pickup_hour],
            'pickup_dayofweek': [request.pickup_dayofweek],
            'PULocationID': [request.PULocationID],
            'DOLocationID': [request.DOLocationID]
        })
        
        # Predict
        prediction = model.predict(features)[0]
        
        return PredictionResponse(predicted_fare=float(max(0, prediction)))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
