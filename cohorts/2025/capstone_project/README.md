# ML Zoomcamp Capstone Project: Taxi Fare Prediction

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Raw Data   │────▶│  Preprocess  │────▶│  XGBoost    │────▶│  FastAPI    │
│  (NYC TLC)  │     │  Pipeline    │     │  Model      │     │  API        │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                   │
                                                                   ▼
                                                            ┌──────────────┐
                                                            │   Docker     │
                                                            │  Container   │
                                                            └──────────────┘
```

## Project Components

### 1. Data Pipeline
- Downloads NYC Yellow Taxi data
- Feature engineering
- Train/test split

### 2. ML Model
- XGBoost Regressor
- Hyperparameter tuning
- Model evaluation (RMSE, MAE, R²)

### 3. Model Deployment
- FastAPI REST API
- Docker containerization
- Health checks

### 4. Prediction Endpoint
- POST /predict
- Input: pickup/dropoff coordinates, passenger count, etc.
- Output: Predicted fare amount

## Execution

```bash
# 1. Create venv
uv venv --python 3.11
source .venv/bin/activate

# 2. Install dependencies
uv pip install xgboost fastapi uvicorn scikit-learn pandas

# 3. Train model
python3 train_model.py

# 4. Start API
uvicorn app:app --host 0.0.0.0 --port 8000

# 5. Test prediction
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"pickup_lat": 40.7128, "pickup_lon": -74.0060, "dropoff_lat": 40.7580, "dropoff_lon": -73.9855, "passenger_count": 2}'
```

## Status: ✅ BUILT & EXECUTED
