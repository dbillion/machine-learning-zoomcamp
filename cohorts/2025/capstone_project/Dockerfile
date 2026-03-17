FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir xgboost fastapi uvicorn scikit-learn pandas pydantic

COPY taxi_fare_model.pkl .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
