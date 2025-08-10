from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def calculate_all(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted, squared=False)
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1))) * 100
    total_actual = actual.sum()
    total_predicted = predicted.sum()
    total_accuracy_pct = (total_actual / total_predicted * 100) if total_predicted > 0 else 0
    correlation = np.corrcoef(actual, predicted)[0, 1] if len(actual) > 1 else 0

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "Total_Actual": total_actual,
        "Total_Predicted": total_predicted,
        "Total_Accuracy_Pct": total_accuracy_pct,
        "Correlation": correlation
    }
