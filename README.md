# Beijing AQI Forecaster 🌫️

Forecast Beijing PM2.5 air quality using Facebook Prophet trained on 5 years of hourly data.

**[▶ Live Demo](https://huggingface.co/spaces/Riochi511/aqi-forecaster)**

---

## Overview

This project builds a time-series forecaster for Beijing PM2.5 air quality (μg/m³) using Facebook Prophet. Given a number of forecast days, the app returns hourly PM2.5 predictions starting from the end of the training period.

PM2.5 measures fine particulate matter — a key air quality indicator with serious health implications. Beijing PM2.5 is known for extreme spikes (400–900 μg/m³), making it a challenging but realistic forecasting target.

---

## Dataset

**Beijing PM2.5 (PRSA Dataset)**

- Source: [Kaggle — Beijing PM2.5 Data](https://www.kaggle.com/datasets/ruiqurm/lifeexpectancy)
- Coverage: January 2010 – December 2014 (hourly readings)
- Raw rows: 43,824
- After cleaning (dropped NaN PM2.5 rows): 41,757 rows
- Features used: datetime index + PM2.5 target only

---

## Model

**Facebook Prophet** with full seasonality:

- Daily seasonality
- Weekly seasonality
- Yearly seasonality

Trained on the full cleaned dataset. Evaluated on a 30-day holdout (last 720 hours).

---

## Evaluation

| Metric | Value |
|--------|-------|
| MAE | 86.27 μg/m³ |
| RMSE | 101.74 μg/m³ |

**Note on error magnitude:** These numbers are high in absolute terms but expected given the nature of PM2.5 data. Beijing PM2.5 regularly spikes to 400–900 μg/m³ during pollution events. Prophet captures seasonal trends well but cannot predict sudden pollution spikes, which are driven by weather inversions and industrial activity — factors not included in this model. Error is framed honestly in the app interface.

---

## App

Built with **Gradio** and deployed on **Hugging Face Spaces**.

- Input: number of forecast days (1–7)
- Output: hourly PM2.5 forecast table with predicted value, lower bound, and upper bound
- Lower bound values are clipped to 0 (PM2.5 cannot be negative)
- Model is downloaded from Google Drive at cold start via `gdown`

---

## Project Structure

```
aqi-forecaster/
├── app.py              # Gradio interface + Prophet inference
├── requirements.txt    # prophet, pandas, numpy, gdown
├── Dockerfile          # Present in repo (unused — HF Gradio SDK handles runtime)
└── README.md
```

`model.pkl` is stored on Google Drive and downloaded at runtime. It is not committed to git due to file size.

---

## Stack

| Layer | Tool |
|-------|------|
| Training | Google Colab |
| Model | Facebook Prophet |
| App framework | Gradio |
| Deployment | Hugging Face Spaces |
| Version control | GitHub |

---

## Limitations

- Prophet captures trend and seasonality but cannot model sudden pollution spikes
- No weather features (wind speed, temperature, humidity, pressure) — major drivers of PM2.5 variability
- Training data ends in 2014; forecasts extrapolate beyond the training window
- Free HF CPU tier may cause slow cold-start inference

---

## Future Improvements

- **LSTM model** with weather features (wind speed, humidity, temperature) for spike prediction
- Retrain on more recent data (post-2014)
- Add WHO guideline threshold annotations to forecast output (WHO safe limit: 15 μg/m³ daily average)

---

## Part of

This project is part of a 12-week ML portfolio roadmap — Week 5–8 milestone.

[GitHub Portfolio](https://github.com/Riochi511) · [Hugging Face](https://huggingface.co/Riochi511)
