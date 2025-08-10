import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class LastMonthForecaster:
    def __init__(self, drug_name, target_metric='total_dose', model_type='auto'):
        self.drug_name = drug_name
        self.target_metric = target_metric
        self.model_type = model_type
        self.selected_model = None
        self.trained_model = None
        self.drug_profile = None
        self.prepared_data = None

    def merge_dataframes(self, *drug_dfs, event_dfs=None):
        print(f"📊 MERGING {len(drug_dfs)} DRUG DATAFRAMES...")
        merged_drug_df = pd.concat(drug_dfs, ignore_index=True)
        print(f"Combined drug records: {len(merged_drug_df)}")

        merged_event_df = None
        if event_dfs is not None:
            merged_event_df = pd.concat(event_dfs, ignore_index=True)
            print(f"Combined event records: {len(merged_event_df)}")

        merged_drug_df['execution_date'] = pd.to_datetime(merged_drug_df['execution_date'])
        date_range = f"{merged_drug_df['execution_date'].min().date()} to {merged_drug_df['execution_date'].max().date()}"
        print(f"Combined date range: {date_range}")

        return merged_drug_df, merged_event_df

    def prepare_data(self, drug_df, event_df=None, filter_zero_dose=True):
        print(f"\n📈 PREPARING DATA FOR {self.drug_name}...")
        drug_data = drug_df[drug_df['mnn'] == self.drug_name].copy()
        print(f"Drug records found: {len(drug_data)}")

        if len(drug_data) == 0:
            raise ValueError(f"No data found for drug: {self.drug_name}")

        if filter_zero_dose:
            initial_count = len(drug_data)
            drug_data = drug_data[(drug_data['dose'].notna()) & (drug_data['dose'] > 0)]
            removed_count = initial_count - len(drug_data)
            print(f"Removed {removed_count} zero/null dose records ({removed_count/initial_count*100:.1f}%)")

        drug_data['date'] = pd.to_datetime(drug_data['execution_date'])
        drug_data = drug_data[drug_data['date'].notna()].sort_values('date')
        print(f"Valid records: {len(drug_data)}")

        daily_consumption = drug_data.groupby(drug_data['date'].dt.date).agg({
            'dose': ['count', 'sum', 'mean', 'std', 'min', 'max'],
            'client_id': 'nunique',
            'event_id': 'nunique',
            'status': [
                lambda x: (x == 2).sum(),
                lambda x: (x == 6).sum(),
                lambda x: (x == 1).sum()
            ]
        }).reset_index()

        daily_consumption.columns = [
            'date', 'count', 'total_dose', 'avg_dose', 'dose_std', 'dose_min', 'dose_max',
            'unique_patients', 'unique_events', 'completed', 'cancelled', 'ready'
        ]
        daily_consumption['date'] = pd.to_datetime(daily_consumption['date'])

        date_range = pd.date_range(
            start=daily_consumption['date'].min(),
            end=daily_consumption['date'].max(),
            freq='D'
        )
        daily_consumption = daily_consumption.set_index('date').reindex(date_range).fillna(0)
        daily_consumption = daily_consumption.reset_index()
        daily_consumption.columns = ['date'] + list(daily_consumption.columns[1:])
        daily_consumption['completion_rate'] = daily_consumption['completed'] / (daily_consumption['count'] + 1)
        daily_consumption['cancellation_rate'] = daily_consumption['cancelled'] / (daily_consumption['count'] + 1)

        self.prepared_data = daily_consumption
        print(f"Prepared dataset: {len(daily_consumption)} days")
        print(f"Date range: {daily_consumption['date'].min().date()} to {daily_consumption['date'].max().date()}")
        print(f"Target metric range: {daily_consumption[self.target_metric].min():.1f} - {daily_consumption[self.target_metric].max():.1f}")
        return daily_consumption

    def analyze_drug_profile(self):
        if self.prepared_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        target_values = self.prepared_data[self.target_metric]
        total_days = len(target_values)
        zero_days = (target_values == 0).sum()
        zero_percentage = zero_days / total_days * 100
        daily_mean = target_values.mean()
        daily_std = target_values.std()
        cv = daily_std / (daily_mean + 1)

        self.drug_profile = {
            'total_days': total_days,
            'zero_percentage': zero_percentage,
            'daily_mean': daily_mean,
            'daily_std': daily_std,
            'coefficient_variation': cv
        }

        print(f"\n📋 DRUG PROFILE:")
        print(f"Total days: {total_days}")
        print(f"Zero days: {zero_days} ({zero_percentage:.1f}%)")
        print(f"Daily mean: {daily_mean:.1f}")
        print(f"Coefficient of variation: {cv:.2f}")
        return self.drug_profile

    def select_model(self):
        if self.model_type != 'auto':
            self.selected_model = self.model_type
            print(f"Manual model selection: {self.selected_model.upper()}")
            return self.selected_model

        if self.drug_profile is None:
            self.analyze_drug_profile()

        profile = self.drug_profile
        zero_pct = profile['zero_percentage']
        daily_mean = profile['daily_mean']
        cv = profile['coefficient_variation']

        if zero_pct < 20 and daily_mean > 1000:
            self.selected_model = 'xgboost'
            reason = f"High-volume regular usage"
        elif zero_pct > 50 and cv > 2:
            self.selected_model = 'arima'
            reason = f"Intermittent pattern"
        elif zero_pct < 30:
            self.selected_model = 'xgboost'
            reason = f"Regular usage pattern"
        else:
            self.selected_model = 'arima'
            reason = f"Temporal pattern detection needed"

        print(f"\n🤖 INTELLIGENT MODEL SELECTION:")
        print(f"Selected model: {self.selected_model.upper()}")
        print(f"Reason: {reason}")
        return self.selected_model

    def prepare_last_month_split(self, days_to_predict=30, max_training_days=None):
        if self.prepared_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        df = self.prepared_data.copy()
        test_start_point = len(df) - days_to_predict
        if test_start_point <= 0:
            raise ValueError(f"Dataset too small.")

        if max_training_days is None:
            train_start_point = 0
        else:
            train_start_point = max(0, test_start_point - max_training_days)

        train_data = df[train_start_point:test_start_point].copy()
        test_data = df[test_start_point:].copy()

        print(f"\n📅 TRAINING/TEST SPLIT:")
        print(f"Training period: {train_data['date'].min().date()} to {train_data['date'].max().date()} ({len(train_data)} days)")
        print(f"Test period: {test_data['date'].min().date()} to {test_data['date'].max().date()} ({len(test_data)} days)")
        return train_data, test_data

    # train_xgboost_model(), train_arima_model(), predict_last_month(), etc.
    # ➡ Keep exactly as in your original code

    def plot_forecast_vs_actual_and_cumulative(self, test_data, predictions, metrics):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{self.drug_name} - Last Month Prediction Analysis ({self.selected_model.upper()})',
                     fontsize=14, fontweight='bold')

        actual_values = np.array(test_data[self.target_metric].values)
        predictions = np.array(predictions)
        dates = pd.to_datetime(test_data['date']).values

        ax1 = axes[0]
        ax1.plot(dates, actual_values, label='Actual', marker='o')
        ax1.plot(dates, predictions, label=f'{self.selected_model.upper()} Forecast', marker='s')
        ax1.set_title('Forecast vs Actual Time Series')
        ax1.legend()

        ax2 = axes[1]
        ax2.plot(dates, np.cumsum(actual_values), label='Cumulative Actual', marker='o')
        ax2.plot(dates, np.cumsum(predictions), label=f'Cumulative {self.selected_model.upper()}', marker='s')
        ax2.set_title('Cumulative Comparison')
        ax2.legend()

        plt.tight_layout()
        return fig


def forecast_last_month_multiple_data(drug_name, drug_dfs, event_dfs=None, days_to_predict=30,
                                      target_metric='total_dose', model_type='auto', filter_zero_dose=True,
                                      max_training_days=None):
    forecaster = LastMonthForecaster(drug_name, target_metric, model_type)
    merged_drug_df, merged_event_df = forecaster.merge_dataframes(*drug_dfs, event_dfs=event_dfs)
    forecaster.prepare_data(merged_drug_df, merged_event_df, filter_zero_dose=filter_zero_dose)
    forecaster.analyze_drug_profile()
    forecaster.select_model()
    train_data, test_data = forecaster.prepare_last_month_split(days_to_predict, max_training_days)

    feature_cols = None
    if forecaster.selected_model == 'xgboost':
        feature_cols = forecaster.train_xgboost_model(train_data)
    elif forecaster.selected_model == 'arima':
        forecaster.train_arima_model(train_data)

    predictions = forecaster.predict_last_month(train_data, test_data, feature_cols)
    metrics = forecaster.calculate_metrics(test_data[target_metric].values, predictions)

    return forecaster, metrics, predictions, test_data


def quick_last_month_forecast(drug_name, days_to_predict=30, max_training_days=None):
    try:
        drug_df_2024 = pd.read_csv('drug_2024_sample_4k.csv')
        drug_df_2025 = pd.read_csv('drug_2025_sample_4k.csv')
        event_df_2024 = pd.read_csv('event_2024_anonim_sample_4k.csv')
        event_df_2025 = pd.read_csv('event_2025_anonim_sample_4k.csv')
        return forecast_last_month_multiple_data(
            drug_name,
            drug_dfs=[drug_df_2024, drug_df_2025],
            event_dfs=[event_df_2024, event_df_2025],
            days_to_predict=days_to_predict,
            max_training_days=max_training_days
        )
    except FileNotFoundError:
        print("❌ Sample data files not found.")
        return None
