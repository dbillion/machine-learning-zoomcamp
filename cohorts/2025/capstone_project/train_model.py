"""
ML Zoomcamp Capstone: Train Taxi Fare Prediction Model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import json

print("=== ML ZOOMCAMP CAPSTONE: MODEL TRAINING ===\n")

# Load data
url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
print(f"Loading data from: {url}")

# Read with pandas (sample for speed)
df = pd.read_parquet(url)
df = df.sample(50000, random_state=42)  # Sample 50K for training
print(f"✅ Loaded {len(df):,} records")

# Feature engineering
print("\nEngineering features...")
df = df[df['fare_amount'] > 0]  # Remove invalid fares
df = df[df['trip_distance'] > 0]  # Remove zero distances
df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]  # Valid passengers

# Extract time features
df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['pickup_hour'] = df['pickup_datetime'].dt.hour
df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
df['trip_distance_km'] = df['trip_distance'] * 1.60934  # Convert to km

# Select features
features = ['passenger_count', 'trip_distance_km', 'pickup_hour', 'pickup_dayofweek', 'PULocationID', 'DOLocationID']
target = 'fare_amount'

X = df[features].fillna(0)
y = df[target]

print(f"Features: {features}")
print(f"Target: {target}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# Train XGBoost model
print("\nTraining XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Model trained")

# Evaluate
print("\n=== EVALUATION ===")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:.2f}")
print(f"MAE: ${mae:.2f}")
print(f"R²: {r2:.4f}")

# Feature importance
print("\n=== FEATURE IMPORTANCE ===")
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
for _, row in importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save model
with open('taxi_fare_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n✅ Model saved: taxi_fare_model.pkl")

# Save metadata
metadata = {
    'model': 'XGBoost Regressor',
    'n_estimators': 100,
    'max_depth': 6,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'rmse': float(rmse),
    'mae': float(mae),
    'r2': float(r2),
    'features': features,
    'target': target
}
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✅ Metadata saved: model_metadata.json")

print("\n=== TRAINING COMPLETE ✅ ===")
