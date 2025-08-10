import os
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np


@dataclass
class InferenceConfig:
    drug_paths: List[str]
    horizon_days: int = 30
    lookback_days: int = 180
    model_type: str = "auto"          # 'auto' | 'xgboost' | 'arima'
    target_metric: str = "total_dose"
    drug_name: Optional[str] = None
    columns_map: Optional[Dict[str, str]] = None
    output_dir: str = "runs"


COMMON_DATE_COLS = ["date", "Дата", "дата", "DAY", "Day", "DATE"]
COMMON_DOSE_COLS = ["total_dose", "dose", "quantity", "Количество", "количество", "sum_dose"]
COMMON_DRUG_COLS = ["drug_name", "Препарат", "drug", "name", "Наименование"]


def _first_existing(d: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in d.columns:
            return c
    return None


def _standardize_columns(df: pd.DataFrame, columns_map: Optional[Dict[str, str]]) -> Tuple[str, str, Optional[str]]:
    if columns_map:
        date_col = columns_map.get("date")
        dose_col = columns_map.get("dose") or columns_map.get("target") or columns_map.get("y")
        drug_col = columns_map.get("drug")
    else:
        date_col = _first_existing(df, COMMON_DATE_COLS)
        dose_col = _first_existing(df, COMMON_DOSE_COLS)
        drug_col = _first_existing(df, COMMON_DRUG_COLS)

    if not date_col or not dose_col:
        raise ValueError(f"Could not detect 'date' and/or 'dose' columns. Got map={columns_map}, df.columns={list(df.columns)[:20]}")
    return date_col, dose_col, drug_col


def _load_union_csv(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {p}")
        frames.append(pd.read_csv(p))
    if not frames:
        raise ValueError("No CSVs loaded")
    return pd.concat(frames, ignore_index=True)


def _prepare_series(df: pd.DataFrame, date_col: str, dose_col: str, drug_col: Optional[str], drug_name: Optional[str], lookback_days: int) -> pd.Series:
    if drug_name and drug_col and drug_col in df.columns:
        df = df[df[drug_col].astype(str) == str(drug_name)].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[dose_col] = pd.to_numeric(df[dose_col], errors="coerce").fillna(0)

    daily = df.groupby(df[date_col].dt.date)[dose_col].sum().rename("y").to_frame()
    daily.index = pd.to_datetime(daily.index)

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index).fillna(0.0)

    if lookback_days is not None and lookback_days > 0:
        cutoff = daily.index.max() - pd.Timedelta(days=lookback_days-1)
        daily = daily.loc[daily.index >= cutoff]
    return daily["y"]


def _select_model_auto(y: pd.Series) -> str:
    if len(y) < 60:
        return "xgboost"
    zero_ratio = (y == 0).mean()
    if zero_ratio >= 0.40:
        return "xgboost"
    return "arima"


def _fit_predict_arima(y: pd.Series, horizon: int) -> np.ndarray:
    from statsmodels.tsa.arima.model import ARIMA
    candidate_orders = [(1,1,1), (2,1,2), (1,1,0), (0,1,1)]
    best_aic = np.inf
    best_model = None
    for order in candidate_orders:
        try:
            res = ARIMA(y, order=order).fit()
            if hasattr(res, "aic") and res.aic < best_aic:
                best_aic = res.aic
                best_model = res
        except Exception:
            continue
    if best_model is None:
        return np.array([float(y.iloc[-1])] * horizon, dtype=float)
    return np.asarray(best_model.forecast(steps=horizon), dtype=float)


def _make_features(x_index: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=x_index)
    f["dow"] = x_index.dayofweek
    f["dom"] = x_index.day
    f["month"] = x_index.month
    f["is_month_start"] = x_index.is_month_start.astype(int)
    f["is_month_end"] = x_index.is_month_end.astype(int)
    return f


def _fit_predict_xgb(y: pd.Series, horizon: int) -> np.ndarray:
    try:
        import xgboost as xgb
    except Exception:
        hist = y.values.astype(float)
        if len(hist) >= 7:
            last_week = hist[-7:]
            return np.tile(last_week, int(np.ceil(horizon / 7)))[:horizon]
        return np.array([hist[-1]] * horizon, dtype=float)

    idx = y.index
    df = pd.DataFrame({"y": y.values}, index=idx)
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for win in [7, 14, 28]:
        df[f"rollmean_{win}"] = df["y"].rolling(win).mean()
        df[f"rollstd_{win}"] = df["y"].rolling(win).std()

    df = df.join(_make_features(idx)).dropna()
    X_train = df.drop(columns=["y"]).values
    y_train = df["y"].values

    if len(y_train) < 20:
        if len(y) >= 7:
            last_week = y.values[-7:]
            return np.tile(last_week, int(np.ceil(horizon / 7)))[:horizon]
        return np.array([y.values[-1]] * horizon, dtype=float)

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=0,
    )
    model.fit(X_train, y_train)

    future_index = pd.date_range(idx.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    hist_df = pd.DataFrame({"y": y.values}, index=idx)
    preds = []
    for cur_date in future_index:
        ext_idx = pd.DatetimeIndex(sorted(hist_df.index.tolist() + [cur_date]))
        extended = hist_df.reindex(ext_idx)
        for lag in [1, 7, 14, 28]:
            extended[f"lag_{lag}"] = extended["y"].shift(lag)
        for win in [7, 14, 28]:
            extended[f"rollmean_{win}"] = extended["y"].rolling(win).mean()
            extended[f"rollstd_{win}"] = extended["y"].rolling(win).std()
        feat = _make_features(pd.DatetimeIndex([cur_date]))
        row = pd.concat([extended.iloc[-1:].drop(columns=["y"]), feat], axis=1)
        train_cols = [c for c in df.columns if c != "y"]
        for c in train_cols:
            if c not in row.columns:
                row[c] = 0
        row = row[train_cols]
        yhat = float(model.predict(row.values)[0])
        preds.append(yhat)
        hist_df.loc[cur_date, "y"] = yhat
    return np.array(preds, dtype=float)


def forecast_from_csv(drug_csv_paths: List[str], horizon_days: int = 30, lookback_days: int = 180,
                      model_type: str = "auto", target_metric: str = "total_dose", drug_name: Optional[str] = None,
                      columns_map: Optional[Dict[str, str]] = None, output_dir: str = "runs") -> pd.DataFrame:
    raw = _load_union_csv(drug_csv_paths)
    date_col, dose_col, drug_col = _standardize_columns(raw, columns_map or {"date": None, "dose": target_metric, "drug": None})
    y = _prepare_series(raw, date_col, dose_col, drug_col, drug_name, lookback_days)
    m = model_type if model_type != "auto" else _select_model_auto(y)
    preds = _fit_predict_arima(y, horizon_days) if m == "arima" else _fit_predict_xgb(y, horizon_days)

    hist_df = pd.DataFrame({"date": y.index, "y_true": y.values, "y_pred": np.nan, "split": "history"})
    future_index = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    fc_df = pd.DataFrame({"date": future_index, "y_true": np.nan, "y_pred": preds, "split": "forecast"})
    out = pd.concat([hist_df, fc_df], ignore_index=True)

    slug = (drug_name or "ALL").replace(" ", "_")
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{stamp}_{slug}")
    os.makedirs(run_dir, exist_ok=True)
    out.to_csv(os.path.join(run_dir, "forecast.csv"), index=False)

    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(hist_df["date"], hist_df["y_true"], label="History")
        plt.plot(fc_df["date"], fc_df["y_pred"], label="Forecast")
        plt.xlabel("Date")
        plt.ylabel(target_metric)
        plt.title(f"Forecast ({m}) — {drug_name or 'ALL DRUGS'}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "forecast.png"), dpi=140)
        plt.close()
    except Exception:
        pass

    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": m,
            "horizon_days": horizon_days,
            "lookback_days": lookback_days,
            "target_metric": target_metric,
            "drug_name": drug_name,
            "input_files": drug_csv_paths,
        }, f, ensure_ascii=False, indent=2)
    return out


def cli():
    import argparse
    p = argparse.ArgumentParser(description="Forecast future drug usage from CSVs")
    p.add_argument("--drug-paths", nargs="+", required=True)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--lookback", type=int, default=180)
    p.add_argument("--model", type=str, default="auto", choices=["auto","xgboost","arima"])
    p.add_argument("--target", type=str, default="total_dose")
    p.add_argument("--drug-name", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="runs")
    args = p.parse_args()

    df = forecast_from_csv(
        drug_csv_paths=args.drug_paths,
        horizon_days=args.horizon,
        lookback_days=args.lookback,
        model_type=args.model,
        target_metric=args.target,
        drug_name=args.drug_name,
        columns_map=None,
        output_dir=args.output_dir,
    )
    print(df.tail(10).to_string(index=False))


if __name__ == "__main__":
    cli()
