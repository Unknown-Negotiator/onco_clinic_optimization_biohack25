import matplotlib.pyplot as plt
import numpy as np
import os

def save_forecast_plots(test_data, predictions, model_name, target_metric, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    actual_values = np.array(test_data[target_metric].values)
    predictions = np.array(predictions)
    dates = test_data['date']

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Forecast vs Actual
    axes[0].plot(dates, actual_values, label='Actual', marker='o')
    axes[0].plot(dates, predictions, label=f'{model_name} Forecast', marker='s')
    axes[0].set_title('Forecast vs Actual')
    axes[0].legend()

    # Plot 2: Cumulative
    axes[1].plot(dates, np.cumsum(actual_values), label='Cumulative Actual', marker='o')
    axes[1].plot(dates, np.cumsum(predictions), label=f'Cumulative {model_name}', marker='s')
    axes[1].set_title('Cumulative')
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "forecast_plots.png"))
    plt.close(fig)
