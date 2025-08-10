# src/forecaster.py
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta

# --- Plot style (same as Colab) ---
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class ForecastConfig:
    drug_name: str
    target_metric: str = "total_dose"
    model_type: str = "auto"        # 'auto' | 'xgboost' | 'arima'
    days_to_predict: int = 30
    max_training_days: Optional[int] = None
    filter_zero_dose: bool = True


class LastMonthForecaster:
    """
    Forecaster that predicts the last N days (default = last month) from all available data.
    Fully matches the behavior of the working Colab script.
    """

    def __init__(self, drug_name: str, target_metric: str = 'total_dose', model_type: str = 'auto'):
        self.drug_name = drug_name
        self.target_metric = target_metric
        self.model_type = model_type

        self.selected_model: Optional[str] = None
        self.trained_model: Optional[Any] = None
        self.drug_profile: Optional[Dict[str, float]] = None
        self.prepared_data: Optional[pd.DataFrame] = None

    # ----------------------------
    # Data merging / preparation
    # ----------------------------
    def merge_dataframes(
        self,
        *drug_dfs: pd.DataFrame,
        event_dfs: Optional[List[pd.DataFrame]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Merge multiple drug dataframes chronologically (and events if provided)."""
        print(f"📊 MERGING {len(drug_dfs)} DRUG DATAFRAMES...")
        merged_drug_df = pd.concat(drug_dfs, ignore_index=True)
        print(f"Combined drug records: {len(merged_drug_df)}")

        merged_event_df = None
        if event_dfs is not None and len(event_dfs) > 0:
            merged_event_df = pd.concat(event_dfs, ignore_index=True)
            print(f"Combined event records: {len(merged_event_df)}")

        merged_drug_df['execution_date'] = pd.to_datetime(merged_drug_df['execution_date'])
        date_range = f"{merged_drug_df['execution_date'].min().date()} → {merged_drug_df['execution_date'].max().date()}"
        print(f"Combined date range: {date_range}")

        return merged_drug_df, merged_event_df

    def prepare_data(
        self,
        drug_df: pd.DataFrame,
        event_df: Optional[pd.DataFrame] = None,
        filter_zero_dose: bool = True
    ) -> pd.DataFrame:
        """Filter by drug, aggregate to daily series, fill gaps, and build derived metrics."""
        print(f"\n📈 PREPARING DATA FOR {self.drug_name}...")

        # Filter drug
        drug_data = drug_df[drug_df['mnn'] == self.drug_name].copy()
        print(f"Drug records found: {len(drug_data)}")
        if len(drug_data) == 0:
            raise ValueError(f"No data found for drug: {self.drug_name}")

        # Optional: remove zero / null dose
        if filter_zero_dose:
            initial_count = len(drug_data)
            drug_data = drug_data[(drug_data['dose'].notna()) & (drug_data['dose'] > 0)]
            removed_count = initial_count - len(drug_data)
            pct = (removed_count / max(initial_count, 1)) * 100
            print(f"Removed {removed_count} zero/null dose records ({pct:.1f}%)")

        # Dates
        drug_data['date'] = pd.to_datetime(drug_data['execution_date'])
        drug_data = drug_data[drug_data['date'].notna()].sort_values('date')
        print(f"Valid records after cleaning: {len(drug_data)}")

        # Aggregate to daily
        daily = drug_data.groupby(drug_data['date'].dt.date).agg({
            'dose': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'client_id': 'nunique',
            'event_id': 'nunique',
            'status': [
                lambda x: (x == 2).sum(),  # completed
                lambda x: (x == 6).sum(),  # cancelled
                lambda x: (x == 1).sum()   # ready
            ]
        }).reset_index()

        daily.columns = [
            'date', 'count', 'total_dose', 'avg_dose', 'dose_std', 'dose_min', 'dose_max',
            'unique_patients', 'unique_events', 'completed', 'cancelled', 'ready'
        ]
        daily['date'] = pd.to_datetime(daily['date'])

        # Reindex to full daily range
        full_range = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
        daily = daily.set_index('date').reindex(full_range).fillna(0).reset_index()
        daily.columns = ['date'] + list(daily.columns[1:])

        # Derived rates
        daily['completion_rate'] = daily['completed'] / (daily['count'] + 1)
        daily['cancellation_rate'] = daily['cancelled'] / (daily['count'] + 1)

        self.prepared_data = daily

        print(f"Prepared dataset: {len(daily)} days")
        print(f"Date range: {daily['date'].min().date()} → {daily['date'].max().date()}")
        print(f"Target range ({self.target_metric}): {daily[self.target_metric].min():.1f} – {daily[self.target_metric].max():.1f}")

        return daily

    # ----------------------------
    # Profiling & model selection
    # ----------------------------
    def analyze_drug_profile(self) -> Dict[str, float]:
        if self.prepared_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        y = self.prepared_data[self.target_metric]
        total_days = len(y)
        zero_days = int((y == 0).sum())
        zero_pct = zero_days / total_days * 100 if total_days else 0.0
        mean = float(y.mean())
        std = float(y.std())
        cv = std / (mean + 1e-9)

        self.drug_profile = {
            'total_days': total_days,
            'zero_percentage': zero_pct,
            'daily_mean': mean,
            'daily_std': std,
            'coefficient_variation': cv
        }

        print("\n📋 DRUG PROFILE:")
        print(f"Total days: {total_days}")
        print(f"Zero days: {zero_days} ({zero_pct:.1f}%)")
        print(f"Daily mean: {mean:.1f}")
        print(f"Coefficient of variation: {cv:.2f}")
        return self.drug_profile

    def select_model(self) -> str:
        """Choose model (auto rules copied from your working code)."""
        if self.model_type != 'auto':
            self.selected_model = self.model_type
            print(f"Manual model selection: {self.selected_model.upper()}")
            return self.selected_model

        if self.drug_profile is None:
            self.analyze_drug_profile()

        p = self.drug_profile
        zero_pct = p['zero_percentage']
        mean = p['daily_mean']
        cv = p['coefficient_variation']

        if zero_pct < 20 and mean > 1000:
            self.selected_model = 'xgboost'
            reason = f"High-volume regular usage (zeros {zero_pct:.1f}%, mean {mean:.0f})"
        elif zero_pct > 50 and cv > 2:
            self.selected_model = 'arima'
            reason = f"Intermittent pattern (zeros {zero_pct:.1f}%, CV {cv:.2f})"
        elif zero_pct < 30:
            self.selected_model = 'xgboost'
            reason = "Regular usage pattern"
        else:
            self.selected_model = 'arima'
            reason = "Temporal pattern detection needed"

        print("\n🤖 INTELLIGENT MODEL SELECTION:")
        print(f"Selected model: {self.selected_model.upper()}")
        print(f"Reason: {reason}")
        return self.selected_model

    # ----------------------------
    # Train / split
    # ----------------------------
    def prepare_last_month_split(
        self,
        days_to_predict: int = 30,
        max_training_days: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.prepared_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        df = self.prepared_data.copy()
        test_start = len(df) - days_to_predict
        if test_start <= 0:
            raise ValueError(f"Dataset too small: need > {days_to_predict} days, have {len(df)}")

        if max_training_days is None:
            train_start = 0
            window_note = "all available data"
        else:
            train_start = max(0, test_start - max_training_days)
            actual_days = test_start - train_start
            window_note = f"last {actual_days} days of available data"

        train = df.iloc[train_start:test_start].copy()
        test = df.iloc[test_start:].copy()

        print("\n📅 TRAINING/TEST SPLIT:")
        print(f"Total: {len(df)} days ({df['date'].min().date()} → {df['date'].max().date()})")
        print(f"Train : {train['date'].min().date()} → {train['date'].max().date()} ({len(train)} days)")
        print(f"Window: {window_note}")
        print(f"Test  : {test['date'].min().date()} → {test['date'].max().date()} ({len(test)} days)")
        print(f"Test target sum: {test[self.target_metric].sum():.1f}")

        return train, test

    def _add_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Feature engineering identical for train and test creation."""
        out = df.copy()
        out['day_of_week'] = out['date'].dt.dayofweek
        out['day_of_month'] = out['date'].dt.day
        out['month'] = out['date'].dt.month
        out['is_weekend'] = (out['day_of_week'] >= 5).astype(int)

        out['day_sin'] = np.sin(2 * np.pi * out['date'].dt.dayofyear / 365.25)
        out['day_cos'] = np.cos(2 * np.pi * out['date'].dt.dayofyear / 365.25)
        out['week_sin'] = np.sin(2 * np.pi * out['day_of_week'] / 7)
        out['week_cos'] = np.cos(2 * np.pi * out['day_of_week'] / 7)

        for lag in [1, 7, 14]:
            out[f'{self.target_metric}_lag_{lag}'] = out[self.target_metric].shift(lag)
        for window in [7, 14]:
            out[f'{self.target_metric}_rolling_mean_{window}'] = (
                out[self.target_metric].rolling(window, min_periods=1).mean()
            )

        out = out.fillna(0)

        drop_cols = {
            'date', 'count', 'total_dose', 'avg_dose', 'dose_std', 'dose_min', 'dose_max',
            'unique_patients', 'unique_events', 'completed', 'cancelled', 'ready',
            'completion_rate', 'cancellation_rate'
        }
        feature_cols = [c for c in out.columns if c not in drop_cols]
        return out, feature_cols

    def train_xgboost_model(self, train_data: pd.DataFrame) -> List[str]:
        print("\n🔧 TRAINING XGBoost MODEL...")
        feat_df, feature_cols = self._add_features(train_data)

        split = int(len(feat_df) * 0.8)
        X_tr = feat_df.iloc[:split][feature_cols]
        y_tr = feat_df.iloc[:split][self.target_metric]
        X_val = feat_df.iloc[split:][feature_cols]
        y_val = feat_df.iloc[split:][self.target_metric]

        self.trained_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.trained_model.fit(X_tr, y_tr)

        # Silent internal validation (no metrics printed)
        _ = np.maximum(self.trained_model.predict(X_val), 0)
        return feature_cols

    def train_arima_model(self, train_data: pd.DataFrame) -> Tuple[int, int, int]:
        print("\n🔧 TRAINING ARIMA MODEL...")
        ts = train_data[self.target_metric]

        # Stationarity
        adf_p = adfuller(ts.dropna())[1]
        needs_diff = adf_p >= 0.05
        d_values = [1] if needs_diff else [0]
        print(f"Stationarity test: p-value={adf_p:.4f}, needs_differencing={needs_diff}")

        best_aic = np.inf
        best_order = (0, d_values[0], 0)

        for p in range(4):
            for d in d_values:
                for q in range(4):
                    try:
                        fitted = ARIMA(ts, order=(p, d, q)).fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except Exception:
                        continue

        print(f"Optimal ARIMA order: {best_order} (AIC: {best_aic:.2f})")
        self.trained_model = ARIMA(ts, order=best_order).fit()
        return best_order

    # ----------------------------
    # Predict
    # ----------------------------
    def predict_last_month(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        print("\n📈 GENERATING PREDICTIONS...")
        if self.selected_model == 'xgboost':
            return self._predict_xgboost(train_data, test_data, feature_cols)
        elif self.selected_model == 'arima':
            return self._predict_arima(test_data)
        else:
            raise ValueError(f"Unknown model type: {self.selected_model}")

    def _predict_xgboost(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        combined = pd.concat([train_data, test_data], ignore_index=True)
        feat_df, _ = self._add_features(combined)
        test_feats = feat_df.tail(len(test_data))[feature_cols]
        preds = np.maximum(self.trained_model.predict(test_feats), 0)

        print(f"XGBoost predictions: min={preds.min():.1f}, max={preds.max():.1f}, mean={preds.mean():.1f}")
        return preds

    def _predict_arima(self, test_data: pd.DataFrame) -> np.ndarray:
        steps = len(test_data)
        preds = np.maximum(np.asarray(self.trained_model.forecast(steps=steps)), 0)
        print(f"ARIMA predictions: min={preds.min():.1f}, max={preds.max():.1f}, mean={preds.mean():.1f}")
        return preds

    # ----------------------------
    # Metrics (minimal)
    # ----------------------------
    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        total_actual = float(np.nansum(actual))
        total_pred = float(np.nansum(predicted))

        if total_actual > 0:
            total_accuracy = 100.0 - (abs(total_pred - total_actual) / total_actual * 100.0)
        else:
            total_accuracy = 0.0

        trend_corr = float(np.corrcoef(actual, predicted)[0, 1]) if len(actual) > 1 else np.nan

        # Exactly four lines:
        print(f"✅ Total Consumption Accuracy: {total_accuracy:.1f}%")
        print(f"✅ Trend Correlation: {trend_corr:.2f}")
        print(f"📈 Total Actual: {total_actual:.0f}")
        print(f"📈 Total Predicted: {total_pred:.0f}")

        return {
            "total_consumption_accuracy": total_accuracy,
            "trend_correlation": trend_corr,
            "total_actual": total_actual,
            "total_predicted": total_pred
        }

    def plot_forecast_vs_actual_and_cumulative(
        self,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        metrics: Dict[str, float]
    ):
        """Two plots: Forecast vs Actual & Cumulative comparison (pandas-safe)."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f'{self.drug_name} - Last Month Prediction Analysis ({self.selected_model.upper()})',
            fontsize=14, fontweight='bold'
        )

        actual_values = np.array(test_data[self.target_metric].values)
        preds = np.array(predictions)
        dates = pd.to_datetime(test_data['date']).values

        # Plot 1
        ax1 = axes[0]
        ax1.plot(dates, actual_values, label='Actual', linewidth=2, marker='o', markersize=4)
        ax1.plot(dates, preds, label=f'{self.selected_model.upper()} Forecast', linewidth=2, marker='s', markersize=4)
        ax1.set_title('Forecast vs Actual Time Series')
        ax1.set_xlabel('Date')
        unit = 'mg' if 'dose' in self.target_metric else 'count'
        ax1.set_ylabel(f'{self.target_metric} ({unit})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2
        ax2 = axes[1]
        ax2.plot(dates, np.cumsum(actual_values), label='Cumulative Actual', linewidth=2, marker='o', markersize=4)
        ax2.plot(dates, np.cumsum(preds), label=f'Cumulative {self.selected_model.upper()}', linewidth=2, marker='s', markersize=4)
        ax2.set_title('Cumulative Comparison')
        ax2.set_xlabel('Date')
        ax2.set_ylabel(f'Cumulative {self.target_metric} ({unit})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        total_text = f"Final Actual: {actual_values.sum():.0f}\nFinal Predicted: {preds.sum():.0f}"
        ax2.text(0.02, 0.98, total_text, transform=ax2.transAxes, va='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.show()
        return fig


# ----------------------------
# Convenience runners
# ----------------------------
def forecast_last_month_multiple_data(
    drug_name: str,
    drug_dfs: List[pd.DataFrame],
    event_dfs: Optional[List[pd.DataFrame]] = None,
    days_to_predict: int = 30,
    target_metric: str = 'total_dose',
    model_type: str = 'auto',
    filter_zero_dose: bool = True,
    max_training_days: Optional[int] = None
):
    """Exact API you used in Colab."""
    print(f"🚀 LAST MONTH FORECASTING FOR {drug_name}")
    print(f"Prediction period: {days_to_predict} days")
    print(f"Target metric: {target_metric}")
    print(f"Model type: {model_type}")
    print(f"Training window: {'All data' if max_training_days is None else f'{max_training_days} days'}")
    print("=" * 60)

    forecaster = LastMonthForecaster(drug_name, target_metric, model_type)

    merged_drug_df, merged_event_df = forecaster.merge_dataframes(
        *drug_dfs, event_dfs=event_dfs
    )
    forecaster.prepare_data(merged_drug_df, merged_event_df, filter_zero_dose=filter_zero_dose)
    forecaster.analyze_drug_profile()
    forecaster.select_model()

    train_data, test_data = forecaster.prepare_last_month_split(
        days_to_predict=days_to_predict,
        max_training_days=max_training_days
    )

    feature_cols = None
    if forecaster.selected_model == 'xgboost':
        feature_cols = forecaster.train_xgboost_model(train_data)
    elif forecaster.selected_model == 'arima':
        forecaster.train_arima_model(train_data)

    preds = forecaster.predict_last_month(train_data, test_data, feature_cols)
    metrics = forecaster.calculate_metrics(test_data[target_metric].values, preds)
    forecaster.plot_forecast_vs_actual_and_cumulative(test_data, preds, metrics)

    print("\n✅ ANALYSIS COMPLETE!")
    print(f"Model used: {forecaster.selected_model.upper()}")
    print(f"Last {days_to_predict} days predicted with {metrics['total_consumption_accuracy']:.1f}% accuracy")

    return forecaster, metrics, preds, test_data


def run_from_config(cfg: Dict[str, Any]):
    """
    Helper to integrate with YAML-driven pipeline.
    Example YAML keys (matches configs/last_month_bleomicin.yaml):
      - drug_name, target_metric, model_type, days_to_predict, max_training_days, filter_zero_dose
      - drug_paths: [path1, path2]
      - event_paths: [path1, path2]
    """
    drug_dfs = [pd.read_csv(p) for p in cfg.get('drug_paths', [])]
    event_paths = cfg.get('event_paths', [])
    event_dfs = [pd.read_csv(p) for p in event_paths] if event_paths else None

    return forecast_last_month_multiple_data(
        drug_name=cfg['drug_name'],
        drug_dfs=drug_dfs,
        event_dfs=event_dfs,
        days_to_predict=cfg.get('days_to_predict', 30),
        target_metric=cfg.get('target_metric', 'total_dose'),
        model_type=cfg.get('model_type', 'auto'),
        filter_zero_dose=cfg.get('filter_zero_dose', True),
        max_training_days=cfg.get('max_training_days', None),
    )
