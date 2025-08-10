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
```text
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
```

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

**DATA USED FOR THIS PROJECT IS PRIVATE, CANNOT BE PUBLICALLY SHARED**

```text
Example:
data/
  drug_2024_sample_4k.csv
  drug_2025_sample_4k.csv
  event_2024_anonim_sample_4k.csv
  event_2025_sample_4k.csv
```

### Run experiment
python main.py configs/last_month.yaml

### Interactive Examples in notebooks/Single_Drug_Forecasting.ipynb

---

## 📁 Outputs
Each run creates a folder under runs/{experiment_name}/{timestamp}/ with:
- metrics.json — performance metrics
- plots/forecast_plots.png — Forecast vs Actual and Cumulative Comparison

---

## 🔮 Inference: Forecast Future Drug Usage

The `src/infer.py` script allows you to quickly forecast drug usage for a specified time horizon based on historical CSV data.

### Usage
python src/infer.py \
    --drug-paths data/drug_2024.csv data/drug_2025.csv \
    --horizon 30 \
    --lookback 180 \
    --date-col date \
    --dose-col total_dose \
    --drug-name "Блеомицин" \
    --drug-col drug_name \
    --output-dir runs

### Parameters
| Parameter       | Type   | Default       | Description |
|-----------------|--------|---------------|-------------|
| --drug-paths    | list   | **required**  | One or more CSV files with historical drug usage. |
| --horizon       | int    | 30            | Number of days to forecast after the last date in the data. |
| --lookback      | int    | 180           | Number of past days to train the model on. |
| --date-col      | str    | "date"        | Name of the date column in your CSVs. |
| --dose-col      | str    | "total_dose"  | Name of the dose/quantity column in your CSVs. |
| --drug-name     | str    | None          | Filter to a specific drug (by name in --drug-col). If omitted, forecasts all drugs combined. |
| --drug-col      | str    | "drug_name"   | Name of the drug column in your CSVs. |
| --output-dir    | str    | "runs"        | Directory to save forecast.csv and forecast.png. |

### Outputs
- forecast.csv — combined history + forecast table:
  date,y_true,y_pred,split
  2024-06-01,120.0,,history
  ...
  2024-12-01,,130.5,forecast

- forecast.png — plot of historical data with forecasted future.

## 🧠 Model Selection Logic
When model_type: auto in the config, the forecaster chooses:
- XGBoost — if high-volume & regular usage (< 20% zero days and mean > 1000) or generally regular (< 30% zero days)
- ARIMA — if usage is sparse/irregular (> 50% zero days and CV > 2) or temporal pattern detection is needed.

## 📄 Data Structure Examples

Below are anonymized previews (first 5 rows) of each dataset used in the project.

---

### **drug_2024_sample_4k.csv** — Drugs prescribed and administered
Contains all prescribed drugs with dose, unit, administration method, and execution status.
```markdown
|      id |   event_id |   client_id | mnn                        | concentration   |   dose | measure   | execution_date       | method            |   status | status_name        |
|--------:|-----------:|------------:|:---------------------------|:----------------|-------:|:----------|:---------------------|:------------------|---------:|:-------------------|
| 1365902 |   20667002 |      437597 | Филграстим                 | nan             |    480 | мг        | 2024-02-02T11:36:16Z | Подкожный         |        2 | выполнено          |
| 1375417 |   20664652 |      131583 | Урсодезоксихолевая кислота | -               |    250 | мг        | nan                  | Для приема внутрь |        6 | отменено           |
| 2542383 |   20712056 |      483596 | Апиксабан                  | nan             |      2 | мг        | 2024-08-09T10:11:52Z | Для приема внутрь |        2 | выполнено          |
| 1468450 |   20667703 |      452603 | Тиамин                     | nan             |     50 | мг        | 2024-02-17T03:31:33Z | Внутривенный      |        2 | выполнено          |
| 1240719 |   20662842 |      451680 | Омепразол                  | nan             |     20 | мг        | nan                  | Для приема внутрь |        1 | готов к выполнению |
```

---

### **event_2024_anonim_sample_4k.csv** — Hospitalization & visit events  
Stores details of each medical event, its type (inpatient, outpatient), and anonymized diagnosis codes.
```markdown
|   event_id |   client_id |   hosp_start |   hosp_end | request_type   |   event_org_code | icd_codes_anon   |
|-----------:|------------:|-------------:|-----------:|:---------------|-----------------:|:-----------------|
|   20673292 |      310694 |          nan |        nan | Поликлиника    |             2.17 | N96, N96         |
|   20716992 |       43630 |          nan |        nan | Поликлиника    |             2.23 | N96, N96         |
|   20733431 |      450059 |          nan |        nan | Поликлиника    |             2.28 | N96, D89         |
|   20704863 |      472770 |          nan |        nan | Поликлиника    |             2.26 | N96, D16         |
|   20687345 |      377357 |          nan |        nan | Поликлиника    |             2.49 | N96, N96         |
```

---

### **instrumental_2024_sample_4k.csv** — Instrumental diagnostics  
Tracks imaging, scans, and other instrumental studies with completion status.
```markdown
|       id |   event_id |   client_id | action_end           |   status | status_name   |   actiontype_id |
|---------:|-----------:|------------:|:---------------------|---------:|:--------------|----------------:|
| 27540010 |   20740432 |      488888 | nan                  |        1 | ожидание      |            4478 |
| 27327848 |   20728854 |      497506 | 2024-10-09T11:18:14Z |        2 | закончено     |            5246 |
| 27134373 |   20713200 |      374378 | nan                  |        2 | закончено     |            4508 |
| 27379994 |   20731988 |      453794 | 2024-10-18T09:33:24Z |        2 | закончено     |            5904 |
| 27397005 |   20731786 |      470642 | nan                  |        2 | закончено     |           12040 |
```

---

### **manipulation_2024_sample_4k.csv** — Medical manipulations & procedures  
Includes therapeutic and diagnostic manipulations, start/end times, and completion status.
```markdown
|       id |   event_id |   client_id | action_end           |   status | status_name   |   actiontype_id |
|---------:|-----------:|------------:|:---------------------|---------:|:--------------|----------------:|
| 27069384 |   20715325 |      246779 | nan                  |        0 | начато        |           12217 |
| 26648798 |   20686748 |      457316 | 2024-05-02T16:00:00Z |        2 | закончено     |           12259 |
| 27137114 |   20718813 |      490840 | 2024-08-24T13:50:31Z |        2 | закончено     |           12259 |
| 26972207 |   20707969 |      441292 | 2024-07-16T13:00:00Z |        2 | закончено     |           12259 |
| 27128136 |   20718597 |      286166 | nan                  |        0 | начато        |            4551 |
```

---

### **lab_2024_sample_4k.csv** — Laboratory tests  
Captures lab analyses, with request and completion timestamps, type of test, and status.
```markdown
|       id |   event_id |   client_id | action_start         | action_end           |   status | status_name   |   actiontype_id |
|---------:|-----------:|------------:|:---------------------|:---------------------|---------:|:--------------|----------------:|
| 26928477 |   20707980 |      106647 | 2024-07-05T11:47:59Z | nan                  |        2 | закончено     |            8439 |
| 27158515 |   20720096 |      436519 | 2024-08-29T12:45:27Z | 2024-08-29T14:18:00Z |        2 | закончено     |            7659 |
| 27604908 |   20742767 |      502052 | 2024-12-03T10:00:44Z | 2024-12-03T11:24:49Z |        2 | закончено     |           12418 |
| 27133276 |   20718940 |      221404 | 2024-08-23T11:16:54Z | 2024-08-30T08:31:32Z |        2 | закончено     |            5096 |
| 26736941 |   20696990 |      476149 | 2024-05-23T13:21:16Z | 2024-05-29T14:42:00Z |        2 | закончено     |           12263 |
```
---

## 🚀 Predicting co-used drugs clusters

**Code for this part of the project is in notebooks/hackathon.ipynb**

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
