import numpy as np

def evaluate_forecast(y_true, y_pred):
    """
    Оценивает прогноз по четырём метрикам:
    - Total Consumption Accuracy
    - Trend Correlation
    - Total Actual
    - Total Predicted
    """

    # Суммарное потребление
    total_actual = np.sum(y_true)
    total_predicted = np.sum(y_pred)

    # Точность по суммарному потреблению
    accuracy = 100 - (abs(total_predicted - total_actual) / total_actual * 100)

    # Корреляция трендов
    if len(y_true) > 1 and len(y_pred) > 1:
        trend_corr = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        trend_corr = np.nan

    # Вывод
    print(f"✅ Total Consumption Accuracy: {accuracy:.1f}%")
    print(f"✅ Trend Correlation: {trend_corr:.2f}")
    print(f"📈 Total Actual: {total_actual:.0f}")
    print(f"📈 Total Predicted: {total_predicted:.0f}")

    return {
        "total_consumption_accuracy": accuracy,
        "trend_correlation": trend_corr,
        "total_actual": total_actual,
        "total_predicted": total_predicted
    }
