---
title: Beijing AQI Forecaster
emoji: 🌫️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
---

# Beijing AQI Forecaster

Forecast Beijing PM2.5 air quality using Facebook Prophet trained on 5 years of hourly data (2010–2014).

**Live demo:** [huggingface.co/spaces/Riochi511/aqi-forecaster](https://huggingface.co/spaces/Riochi511/aqi-forecaster)

## Model
- Algorithm: Facebook Prophet (daily + weekly + yearly seasonality)
- Dataset: Beijing PRSA (41,757 hourly rows, 2010–2014)
- MAE: 86.27 μg/m³ · RMSE: 101.74 μg/m³ on 30-day holdout