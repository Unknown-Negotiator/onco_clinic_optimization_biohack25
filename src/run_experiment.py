import os
import yaml
from datetime import datetime
from .data_loader import load_data
from .forecaster import forecast_last_month_multiple_data
from .plots import save_forecast_plots
import json

def run_experiment(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_dir = os.path.join("runs", cfg["experiment_name"], datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(exp_dir, exist_ok=True)

    drug_dfs, event_dfs = load_data(cfg["drug_paths"], cfg.get("event_paths"))

    forecaster, metrics, predictions, test_data = forecast_last_month_multiple_data(
        cfg["drug_name"], drug_dfs, event_dfs,
        days_to_predict=cfg["days_to_predict"],
        target_metric=cfg["target_metric"],
        model_type=cfg["model_type"],
        filter_zero_dose=cfg["filter_zero_dose"],
        max_training_days=cfg["max_training_days"]
    )

    # Save metrics
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Save plots
    save_forecast_plots(test_data, predictions, forecaster.selected_model.upper(),
                        cfg["target_metric"], os.path.join(exp_dir, "plots"))

    print(f"✅ Run saved in {exp_dir}")
