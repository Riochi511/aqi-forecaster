import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prophet import Prophet

app = FastAPI(title="AQI Forecaster API")

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class ForecastRequest(BaseModel):
    days: int = 7

@app.get("/")
def root():
    return {"message": "AQI Forecaster API is running"}

@app.post("/forecast")
def forecast(request: ForecastRequest):
    future = model.make_future_dataframe(
        periods=request.days * 24, 
        freq='H'
    )
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(
        request.days * 24
    )
    result['yhat'] = np.clip(result['yhat'], 0, None)
    result['yhat_lower'] = np.clip(result['yhat_lower'], 0, None)
    result['ds'] = result['ds'].astype(str)
    return result.to_dict(orient='records')
