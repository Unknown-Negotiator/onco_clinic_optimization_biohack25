import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


def forecast_from_csv(
    csv_paths,
    horizon_days=30,
    lookback_days=180,
    date_col="date",
    dose_col="total_dose",
    drug_name=None,
    drug_col="drug_name",
    output_dir="runs"
):
    # Load all CSVs
    dfs = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    # Filter by drug if requested
    if drug_name and drug_col in df.columns:
        df = df[df[drug_col] == drug_name]

    # Parse date, numeric dose
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce").fillna(0)

    # Aggregate daily totals
    daily = df.groupby(df[date_col].dt.date)[dose_col].sum()
    daily.index = pd.to_datetime(daily.index)

    # Fill missing days
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx, fill_value=0)

    # Keep only lookback window
    cutoff = daily.index.max() - pd.Timedelta(days=lookback_days - 1)
    daily = daily.loc[daily.index >= cutoff]

    # Fit ARIMA (simple, fixed order)
    model = ARIMA(daily, order=(1, 1, 1))
    model_fit = model.fit()

    # Forecast
    forecast_index = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    forecast_values = model_fit.forecast(steps=horizon_days)

    # Build output dataframe
    hist_df = pd.DataFrame({"date": daily.index, "y_true": daily.values, "y_pred": np.nan, "split": "history"})
    fc_df = pd.DataFrame({"date": forecast_index, "y_true": np.nan, "y_pred": forecast_values.values, "split": "forecast"})
    out = pd.concat([hist_df, fc_df], ignore_index=True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "forecast.csv")
    out.to_csv(out_path, index=False)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(hist_df["date"], hist_df["y_true"], label="History")
    plt.plot(fc_df["date"], fc_df["y_pred"], label="Forecast")
    plt.legend()
    plt.title(f"Forecast — {drug_name or 'ALL DRUGS'}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "forecast.png"))
    plt.close()

    print(f"Saved forecast to {out_path}")
    return out


if __name__ == "__main__":
    # Example usage
    df = forecast_from_csv(
        ["drug_2024.csv", "drug_2025.csv"],
        horizon_days=30,
        lookback_days=180,
        date_col="date",
        dose_col="total_dose",
        drug_name="Блеомицин",  # or None
        drug_col="drug_name",
        output_dir="runs"
    )
    print(df.tail(10))
