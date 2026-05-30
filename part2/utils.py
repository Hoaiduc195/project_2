import math

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(X: list) -> list:
    # lib version
    X_array = np.array(X)
    vif_values = [variance_inflation_factor(X_array, i) for i in range(X_array.shape[1])]
    return vif_values

def remove_column(X, col_idx):
    new_X = []
    for row in X:
        new_row = [
            row[j]
            for j in range(len(row))
            if j != col_idx
        ]
        new_X.append(new_row)
    return new_X

def metrics_summary(metrics):
    print("Model Metrics:")
    print(f"R²             : {metrics['R_squared']:.6f}")
    print(f"Adjusted R²    : {metrics['Adj_R_squared']:.6f}")
    print(f"F-statistic    : {metrics['F_stat']:.6f}")
    print(f"F p-value      : {metrics['F_p_value']:.6f}")
    print("\n")

def coef_inference_summary(inference, beta_hat, feature_names):
    headers = [
        "Variable",
        "Coef",
        "Std Err",
        "t",
        "P>|t|",
        "CI Lower",
        "CI Upper"
    ]

    print("{:<15} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(*headers))

    variables = ["Intercept"] + feature_names

    for i in range(len(beta_hat)):

        print(
            "{:<15} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}".format(
                variables[i],
                beta_hat[i],
                inference['se'][i],
                inference['t_stat'][i],
                inference['p_value'][i],
                inference['ci_lower'][i],
                inference['ci_upper'][i]
            )
        )
    
def compute_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))

def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def test_metrics_summary(y_true, y_pred):
    mae = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    r2_value = compute_r2(y_true, y_pred)
    print(f"MAE             : {mae:.6f}")
    print(f"RMSE            : {rmse:.6f}")
    print(f"R²              : {r2_value:.6f}")
