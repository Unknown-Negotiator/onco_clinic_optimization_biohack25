# Onco Clinic Optimization — Biohack 2025

## 📌 Overview
This project forecasts drug consumption, services usage, and hospitalization flow for an oncohematological clinic using real-world data.  
It supports:
- Predicting the last month from any dataset.
- Automatic model selection between XGBoost and ARIMA based on usage patterns.
- Flexible configuration via YAML.
- Modular code structure for adding new models, metrics, or plots.
- Saving all experiment results (metrics + plots) in organized run folders.

---

## 📂 Project Structure
biohack_forecast/
├─ configs/                 # Experiment configs (YAML)
├─ data/                    # CSV data (ignored by Git)
├─ runs/                    # Auto-created experiment outputs
├─ src/
│  ├─ forecaster.py         # LastMonthForecaster class + core logic
│  ├─ data_loader.py        # Data reading helpers
│  ├─ metrics.py            # Metrics calculations
│  ├─ plots.py              # Plotting functions
│  └─ run_experiment.py     # Experiment orchestration
├─ main.py                  # CLI entrypoint
├─ requirements.txt
└─ README.md

---

## ⚙️ Installation

### 1. Clone the repository
git clone git@github.com:Unknown-Negotiator/onco_clinic_optimization_biohack25.git
cd onco_clinic_optimization_biohack25

### 2. Create Conda environment (recommended)
conda create -n biohack_forecast python=3.10 -y
conda activate biohack_forecast

### 3. Install dependencies
pip install -r requirements.txt

---

## 📊 Usage

### Prepare your data
Place your CSV files in data/ or update configs/last_month.yaml paths.

DATA USED FOR THIS PROJECT IS PRIVATE, CANNOT BE PUBLICALLY SHARED

Example:
data/
  drug_2024_sample_4k.csv
  drug_2025_sample_4k.csv
  event_2024_anonim_sample_4k.csv
  event_2025_sample_4k.csv

### Run experiment
python main.py configs/last_month.yaml

---

## 📁 Outputs
Each run creates a folder under runs/{experiment_name}/{timestamp}/ with:
- metrics.json — performance metrics
- plots/forecast_plots.png — Forecast vs Actual and Cumulative Comparison

---

## 🧠 Model Selection Logic
When model_type: auto in the config, the forecaster chooses:
- XGBoost — if high-volume & regular usage (< 20% zero days and mean > 1000) or generally regular (< 30% zero days)
- ARIMA — if usage is sparse/irregular (> 50% zero days and CV > 2) or temporal pattern detection is needed.

---

## 🚀 Predicting co-used drugs clusters


This project analyzes prescription data for forecasting future drug usage volumes.

1. Graph and Cluster Construction

 • Built a drug co-occurrence graph based on joint prescriptions.

 • Strong connections were identified using the Jaccard coefficient.

 • Nodes were grouped into clusters (color-coded in the graph). Each cluster represents stable treatment schemes.

2. Forecasting Models

 • Tested different approaches, including Prophet without external regressors and Prophet with clusters as regressors.

 • The best results were achieved by Prophet with clusters: using cluster dynamics as external regressors reduced forecast errors according to MAE, MAPE, and WAPE metrics.

3. Results

 • The cluster-based model captures trends more accurately and better predicts peaks and drops compared to the model without clusters.

 • This approach enables consideration of complex treatment schemes rather than only aggregated totals.

4. Next Steps

 • Refine clustering: explore alternative community detection algorithms and tune Jaccard thresholds.

 • Feature engineering: incorporate additional clinical and temporal features (e.g., patient demographics, seasonal effects).
 • Hybrid models: test gradient boosting or neural networks with cluster dynamics as inputs.
 • Granularity: build separate forecasts per major therapeutic area and aggregate them for improved accuracy.
 • Continuous learning: implement a rolling retraining pipeline to adapt to new data in near real time.
