import os
import pickle
import numpy as np
import pandas as pd
import gradio as gr
import gdown

# ── Download model from Google Drive on cold start ───────────────────────────
if not os.path.exists("model.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=1OyEqZj3pbNXjG1U3kBgt1l6KW1osPd3e",
        "model.pkl",
        quiet=False
    )

# ── Load model ────────────────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ── Forecast function ─────────────────────────────────────────────────────────
def forecast_aqi(days: int):
    future = model.make_future_dataframe(periods=days * 24, freq='H')
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days * 24)
    result['yhat']       = np.clip(result['yhat'],       0, None).round(1)
    result['yhat_lower'] = np.clip(result['yhat_lower'], 0, None).round(1)
    result['yhat_upper'] = result['yhat_upper'].round(1)
    result['ds']         = result['ds'].dt.strftime('%Y-%m-%d %H:00')
    result.columns       = ['DateTime', 'Forecast PM2.5', 'Lower Bound', 'Upper Bound']
    return result

# ── Custom CSS ────────────────────────────────────────────────────────────────
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@400;700&display=swap');

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #E8F4FD !important;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(160deg, #87CEEB 0%, #B0D9F5 40%, #D6EEF8 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid #A8D8F0;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -40px; left: -60px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.25);
    border-radius: 50%;
    filter: blur(40px);
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: -30px; right: -40px;
    width: 160px; height: 160px;
    background: rgba(255,255,255,0.2);
    border-radius: 50%;
    filter: blur(30px);
}
.app-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    color: #1A3A52 !important;
    margin-bottom: 0.4rem !important;
    position: relative;
}
.app-header h1 span { color: #2980B9; }
.app-header .location {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #2980B9;
    margin-bottom: 0.75rem;
    display: block;
}
.app-header p {
    color: #2C5F7A !important;
    font-size: 0.875rem !important;
    max-width: 520px;
    margin: 0 auto !important;
    line-height: 1.6 !important;
    position: relative;
}
.stat-row {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin-top: 1.75rem;
    flex-wrap: wrap;
    position: relative;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #1A3A52;
    display: block;
    font-family: 'Syne', sans-serif;
}
.stat-label {
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4A7D9A;
}

/* ── AQI legend ── */
.aqi-legend {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 1.25rem;
    font-size: 0.72rem;
    font-weight: 600;
}
.aqi-chip {
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.04em;
}

/* ── Cards ── */
.gradio-container .block,
.gradio-container .form {
    background: rgba(255,255,255,0.85) !important;
    border: 1px solid #BDD9ED !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(41,128,185,0.08) !important;
    backdrop-filter: blur(4px);
}

/* ── Slider ── */
.gradio-container input[type=range] {
    accent-color: #2980B9;
}
.gradio-container .label-wrap span {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    color: #2980B9 !important;
}

/* ── Button ── */
.gradio-container button.primary {
    background: linear-gradient(135deg, #2980B9 0%, #1A6A9A 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1.5rem !important;
    transition: opacity 0.15s !important;
    box-shadow: 0 2px 8px rgba(41,128,185,0.3) !important;
}
.gradio-container button.primary:hover {
    opacity: 0.9 !important;
}

/* ── Table ── */
.gradio-container table {
    font-size: 0.825rem !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container table thead tr th {
    background: #2980B9 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    padding: 0.6rem 1rem !important;
}
.gradio-container table tbody tr:nth-child(even) {
    background: #EAF4FB !important;
}
.gradio-container table tbody tr:hover {
    background: #D6EEF8 !important;
}

/* ── Footer ── */
.footer-note {
    text-align: center;
    font-size: 0.73rem;
    color: #7AAEC8;
    margin-top: 1rem;
    padding: 0.5rem;
    line-height: 1.6;
}
"""

# ── Header HTML ───────────────────────────────────────────────────────────────
header_html = """
<div class="app-header">
  <span class="location">📍 Beijing, China</span>
  <h1>Air Quality <span>Forecast</span></h1>
  <p>
    PM2.5 predictions powered by Facebook Prophet — trained on 5 years of
    hourly readings. Select a horizon and generate your forecast.
  </p>
  <div class="stat-row">
    <div class="stat">
      <span class="stat-value">41,757</span>
      <span class="stat-label">Training rows</span>
    </div>
    <div class="stat">
      <span class="stat-value">86.27</span>
      <span class="stat-label">MAE μg/m³</span>
    </div>
    <div class="stat">
      <span class="stat-value">2010–14</span>
      <span class="stat-label">Training period</span>
    </div>
  </div>
</div>
"""

aqi_legend_html = """
<div class="aqi-legend">
  <span class="aqi-chip" style="background:#D4EDDA;color:#155724;">Good &lt;12</span>
  <span class="aqi-chip" style="background:#FFF3CD;color:#856404;">Moderate 12–35</span>
  <span class="aqi-chip" style="background:#FFE0B2;color:#E65100;">Unhealthy 35–55</span>
  <span class="aqi-chip" style="background:#FFCDD2;color:#B71C1C;">Very Unhealthy &gt;55</span>
</div>
"""

footer_html = """
<div class="footer-note">
  Model: Facebook Prophet &nbsp;·&nbsp; Dataset: Beijing PRSA (2010–2014) &nbsp;·&nbsp;
  High variance reflects PM2.5 spike volatility, a known limitation of trend-based models.<br/>
  Future improvement: LSTM with weather features.
</div>
"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=css, title="Beijing AQI Forecaster") as demo:
    gr.HTML(header_html)
    gr.HTML(aqi_legend_html)

    with gr.Row():
        with gr.Column(scale=1):
            days_slider = gr.Slider(
                minimum=1, maximum=30, value=7, step=1,
                label="Forecast Horizon (Days)"
            )
            run_btn = gr.Button("Generate Forecast", variant="primary")

        with gr.Column(scale=2):
            output_table = gr.Dataframe(
                label="PM2.5 Forecast (μg/m³)",
                headers=["DateTime", "Forecast PM2.5", "Lower Bound", "Upper Bound"],
                wrap=False,
            )

    run_btn.click(fn=forecast_aqi, inputs=days_slider, outputs=output_table)
    gr.HTML(footer_html)

demo.launch()