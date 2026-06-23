import os
import pickle
import numpy as np
import pandas as pd
import gradio as gr
import gdown

# Download model from Google Drive if not present
if not os.path.exists("model.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=1OyEqZj3pbNXjG1U3kBgt1l6KW1osPd3e",
        "model.pkl",
        quiet=False
    )

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def forecast_aqi(days: int):
    future = model.make_future_dataframe(periods=days * 24, freq='H')
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days * 24)
    result['yhat'] = np.clip(result['yhat'], 0, None)
    result['yhat_lower'] = np.clip(result['yhat_lower'], 0, None)
    result['ds'] = result['ds'].dt.strftime('%Y-%m-%d %H:00')
    result.columns = ['DateTime', 'Forecast PM2.5', 'Lower Bound', 'Upper Bound']
    return result

demo = gr.Interface(
    fn=forecast_aqi,
    inputs=gr.Slider(minimum=1, maximum=30, value=7, step=1, label="Forecast Days"),
    outputs=gr.Dataframe(label="PM2.5 Forecast (μg/m³)"),
    title="Beijing AQI Forecaster",
    description="Forecast Beijing PM2.5 air quality using Facebook Prophet trained on 5 years of hourly data (2010-2014). MAE: 86.27 μg/m³ — high variance reflects PM2.5 spike volatility, a known limitation of trend-based models. Future improvement: LSTM with weather features."
)

demo.launch()
