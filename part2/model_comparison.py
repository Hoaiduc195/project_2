import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless plotting to prevent blocking
import matplotlib.pyplot as plt
import scipy.special as sp
from advanced_methods import (
    KernelRidgeRegression,
    BayesianLinearRegression,
    remove_intercept_column,
    tune_krr,
    tune_bayesian_regression,
    validation_split,
)
# Add parent directory to path to ensure we can import from part1
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Ridge, Lasso, and VIF from part1
try:
    from part1.ridge_lasso import vif, ridge_fit, lasso_fit
except ImportError:
    # Local fallback for Ridge/Lasso if import fails (should succeed)
    def vif(X):
        n, p = X.shape
        vifs = np.zeros(p)
        for j in range(p):
            if np.std(X[:, j]) < 1e-9:
                vifs[j] = np.nan
                continue
            y_sub = X[:, j]
            other_indices = [idx for idx in range(p) if idx != j]
            X_sub_features = X[:, other_indices]
            X_sub = np.column_stack((np.ones(n), X_sub_features))
            try:
                beta_sub = np.linalg.solve(X_sub.T @ X_sub, X_sub.T @ y_sub)
                y_hat_sub = X_sub @ beta_sub
                tss = np.sum((y_sub - np.mean(y_sub))**2)
                rss = np.sum((y_sub - y_hat_sub)**2)
                r2 = 0.0 if tss < 1e-9 else 1.0 - (rss / tss)
                vifs[j] = float('inf') if r2 >= 1.0 - 1e-9 else 1.0 / (1.0 - r2)
            except np.linalg.LinAlgError:
                vifs[j] = float('inf')
        return vifs

    def ridge_fit(X, y, lam):
        n, p = X.shape
        I = np.identity(p)
        I[0, 0] = 0.0 # Intercept not penalized
        return np.linalg.solve(X.T @ X + lam * I, X.T @ y)

    def lasso_fit(X, y, lam, max_iter=1000, tol=1e-4):
        n, p = X.shape
        beta = np.zeros(p)
        y_hat = np.zeros(n)
        sum_sq_X = np.sum(X**2, axis=0)
        for iteration in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                sum_sq_Xj = sum_sq_X[j]
                if sum_sq_Xj == 0:
                    beta[j] = 0.0
                    continue
                rho_j = np.dot(X[:, j], y - y_hat) + sum_sq_Xj * beta[j]
                beta_j_old = beta[j]
                if j == 0:
                    beta[j] = rho_j / sum_sq_Xj
                else:
                    lambda_n = lam * n
                    if rho_j < -lambda_n:
                        beta[j] = (rho_j + lambda_n) / sum_sq_Xj
                    elif rho_j > lambda_n:
                        beta[j] = (rho_j - lambda_n) / sum_sq_Xj
                    else:
                        beta[j] = 0.0
                if beta[j] != beta_j_old:
                    y_hat += X[:, j] * (beta[j] - beta_j_old)
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        return beta

# Try to import from teammate's data_pipeline, fallback if empty or error
try:
    from part2.data_pipeline import DataPipeline
    # Verify the imported class has required methods
    dp_test = DataPipeline()
    if not hasattr(dp_test, 'fit') or not hasattr(dp_test, 'transform'):
        raise ImportError
except (ImportError, AttributeError, TypeError):
    # Stateful Processing Pipeline Fallback Implementation
    class DataPipeline:
        """
        Stateful preprocessing pipeline to handle missing value imputation,
        one-hot encoding, and Z-score normalization without data leakage.
        """
        def __init__(self, strategy='median'):
            self.strategy = strategy
            self.num_means = {}
            self.num_stds = {}
            self.num_medians = {}
            self.cat_modes = {}
            self.cat_categories = {}
            self.num_cols = []
            self.cat_cols = []
            self.columns_out_ = []
            
        def fit(self, df, num_cols, cat_cols):
            self.num_cols = list(num_cols)
            self.cat_cols = list(cat_cols)
            
            # Learn numerical metrics
            for col in self.num_cols:
                vals = df[col].dropna()
                if len(vals) > 0:
                    self.num_means[col] = vals.mean()
                    self.num_stds[col] = vals.std(ddof=1)
                    self.num_medians[col] = vals.median()
                else:
                    self.num_means[col] = 0.0
                    self.num_stds[col] = 1.0
                    self.num_medians[col] = 0.0
                    
            # Learn categorical metrics & drop first category to avoid multicollinearity
            for col in self.cat_cols:
                vals = df[col].dropna()
                if len(vals) > 0:
                    self.cat_modes[col] = vals.mode().iloc[0]
                else:
                    self.cat_modes[col] = "Missing"
                
                unique_cats = sorted(list(vals.unique()))
                if len(unique_cats) > 1:
                    self.cat_categories[col] = unique_cats[1:] # Drop first
                else:
                    self.cat_categories[col] = []
                    
            # Set target output columns order
            self.columns_out_ = ['Intercept'] + self.num_cols
            for col in self.cat_cols:
                for cat in self.cat_categories[col]:
                    self.columns_out_.append(f"{col}_{cat}")
            return self
            
        def transform(self, df):
            X_out = pd.DataFrame(index=df.index)
            X_out['Intercept'] = 1.0
            
            # Numerical columns
            for col in self.num_cols:
                series = df[col].copy()
                series.fillna(self.num_medians[col], inplace=True)
                mean = self.num_means[col]
                std = self.num_stds[col]
                if std < 1e-9:
                    std = 1.0
                X_out[col] = (series - mean) / std
                
            # Categorical columns
            for col in self.cat_cols:
                series = df[col].copy()
                series.fillna(self.cat_modes[col], inplace=True)
                for cat in self.cat_categories[col]:
                    X_out[f"{col}_{cat}"] = (series == cat).astype(float)
                    
            return X_out[self.columns_out_].to_numpy(), self.columns_out_

# Try to import OLS and inference from teammate's ols_implementation, fallback if empty
try:
    from part1.ols_implementation import ols_fit, coef_inference
    # Check if they are callable
    if not callable(ols_fit) or not callable(coef_inference):
        raise ImportError
    X_probe = np.column_stack([np.ones(5), np.arange(5, dtype=float)])
    y_probe = np.arange(5, dtype=float)
    ols_probe = ols_fit(X_probe, y_probe)
    if len(ols_probe) != 4 or len(ols_probe[0]) != X_probe.shape[1]:
        raise ImportError
    inference_probe = coef_inference(X_probe, y_probe, ols_probe[0], ols_probe[1])
    if 'p_values' not in inference_probe:
        raise ImportError
except (ImportError, AttributeError):
    # Fallback OLS Implementations
    def ols_fit(X, y):
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        residuals = y - X @ beta
        rss = np.sum(residuals**2)
        n, p = X.shape
        df_err = n - p
        sigma2 = rss / df_err if df_err > 0 else 0.0
        return beta, sigma2, residuals, rss

    def coef_inference(X, y, beta, sigma2):
        n, p = X.shape
        df_err = n - p
        try:
            X_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            X_inv = np.linalg.pinv(X.T @ X)
        cov_beta = sigma2 * X_inv
        
        se = np.zeros(p)
        t_stats = np.zeros(p)
        p_values = np.zeros(p)
        ci_lower = np.zeros(p)
        ci_upper = np.zeros(p)
        t_crit = sp.stdtrit(df_err, 0.975) if df_err > 0 else 1.96
        
        for j in range(p):
            se[j] = np.sqrt(max(0.0, cov_beta[j, j]))
            if se[j] > 0:
                t_stats[j] = beta[j] / se[j]
            else:
                t_stats[j] = 0.0
            p_values[j] = 2.0 * (1.0 - sp.stdtr(df_err, np.abs(t_stats[j]))) if df_err > 0 else 1.0
            ci_lower[j] = beta[j] - t_crit * se[j]
            ci_upper[j] = beta[j] + t_crit * se[j]
            
        return {
            'se': se,
            't_stats': t_stats,
            'p_values': p_values,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

# Evaluation Metric Scoring
def compute_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def compute_r2(y_true, y_pred):
    tss = np.sum((y_true - np.mean(y_true))**2)
    rss = np.sum((y_true - y_pred)**2)
    if tss < 1e-9:
        return 0.0
    return 1.0 - (rss / tss)

# Cross-Validation Lambda Tuner
def kfold_cv_tune(X, y, k=5, lambdas=None, model_type='ridge'):
    np.random.seed(42)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    fold_sizes = np.full(k, n // k)
    fold_sizes[:n % k] += 1
    
    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current:current+size])
        current += size
        
    mse_results = {lam: [] for lam in lambdas}
    
    for fold_idx in range(k):
        test_indices = folds[fold_idx]
        train_indices = np.hstack([folds[i] for i in range(k) if i != fold_idx])
        
        X_tr, y_tr = X[train_indices], y[train_indices]
        X_te, y_te = X[test_indices], y[test_indices]
        
        for lam in lambdas:
            if model_type == 'ridge':
                beta = ridge_fit(X_tr, y_tr, lam)
            elif model_type == 'lasso':
                # Limit max_iter in CV for speed
                beta = lasso_fit(X_tr, y_tr, lam, max_iter=300, tol=1e-4)
            else:
                raise ValueError("Unknown model type")
                
            y_pred = X_te @ beta
            mse = np.mean((y_te - y_pred)**2)
            mse_results[lam].append(mse)
            
    mean_mses = {lam: np.mean(mse_results[lam]) for lam in lambdas}
    best_lam = min(mean_mses, key=mean_mses.get)
    return best_lam, mean_mses

# Feature Selection Routine
def run_feature_selection(X_train, y_train, column_names):
    """
    Perform VIF and p-value feature selection iteratively on training data.
    """
    print("\n[Feature Selection] Starting pruning process...")
    current_indices = list(range(X_train.shape[1]))
    
    # 1. VIF Pruning (multicollinearity filter)
    while True:
        sub_X = X_train[:, current_indices]
        vifs = vif(sub_X)
        
        max_vif = -1.0
        max_idx_in_sub = -1
        
        for i, val in enumerate(vifs):
            if not np.isnan(val) and val > max_vif:
                max_vif = val
                max_idx_in_sub = i
                
        if max_vif > 10.0:
            removed_col_idx = current_indices[max_idx_in_sub]
            print(f"  - Pruned: '{column_names[removed_col_idx]}' due to high VIF ({max_vif:.2f} > 10.0)")
            current_indices.pop(max_idx_in_sub)
        else:
            break
            
    print(f"  - Post-VIF Pruning: {len(current_indices)} columns remaining.")
    
    # 2. p-value Pruning (statistical confidence filter)
    X_vif = X_train[:, current_indices]
    beta_vif, sigma2_vif, _, _ = ols_fit(X_vif, y_train)
    inf_vif = coef_inference(X_vif, y_train, beta_vif, sigma2_vif)
    p_vals = inf_vif['p_values']
    
    final_indices = [current_indices[0]] # Intercept
    for i in range(1, len(current_indices)):
        orig_col_idx = current_indices[i]
        p_val = p_vals[i]
        if p_val <= 0.05:
            final_indices.append(orig_col_idx)
        else:
            print(f"  - Pruned: '{column_names[orig_col_idx]}' due to high p-value ({p_val:.4f} > 0.05)")
            
    print(f"  - Final Feature-Selected Set: {len(final_indices)} columns remaining.")
    return final_indices

# Analytical Diagnostic Visualization Module
def residual_plots(X, y, beta_hat, save_path=None):
    """
    Construct 4 subplots of residuals diagnostics from scratch:
    1. Residuals vs Fitted
    2. Normal Q-Q
    3. Scale-Location
    4. Cook's Distance
    """
    n, p_cols = X.shape
    y_hat = X @ beta_hat
    residuals = y - y_hat
    rss = np.sum(residuals**2)
    df_err = n - p_cols
    sigma2 = rss / df_err if df_err > 0 else 1e-9
    sigma = np.sqrt(sigma2)
    
    # Calculate Hat Matrix Diagonals (Leverage)
    try:
        X_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        X_inv = np.linalg.pinv(X.T @ X)
    H_diag = np.sum((X @ X_inv) * X, axis=1)
    
    # Standardized Residuals
    h_factor = 1.0 - H_diag
    h_factor = np.where(h_factor < 1e-9, 1e-9, h_factor)
    std_eps = residuals / (sigma * np.sqrt(h_factor))
    
    # Cook's Distance
    cooks_d = (std_eps**2 / p_cols) * (H_diag / h_factor)
    
    # Plot Setup
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Regression Diagnostics & Residual Analysis", fontsize=16, fontweight='bold')
    
    # 1. Residuals vs Fitted
    ax1 = axes[0, 0]
    ax1.scatter(y_hat, residuals, alpha=0.5, edgecolors='none', color='#1f77b4')
    ax1.axhline(0, color='red', linestyle='--', lw=1.5)
    # Polynomial trend line
    poly_coefs = np.polyfit(y_hat, residuals, 2)
    fit_x = np.linspace(np.min(y_hat), np.max(y_hat), 100)
    fit_y = np.polyval(poly_coefs, fit_x)
    ax1.plot(fit_x, fit_y, color='darkblue', lw=2, label='Smoothed Trend')
    ax1.set_title("Residuals vs Fitted", fontsize=12, fontweight='semibold')
    ax1.set_xlabel("Fitted Values (Predictions)", fontsize=10)
    ax1.set_ylabel("Residuals", fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend()
    
    # 2. Normal Q-Q
    ax2 = axes[0, 1]
    probs = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = sp.ndtri(probs)
    sorted_std_eps = np.sort(std_eps)
    ax2.scatter(theoretical_quantiles, sorted_std_eps, alpha=0.5, color='#2ca02c')
    
    # Reference Q-Q line
    x25, x75 = np.percentile(theoretical_quantiles, [25, 75])
    y25, y75 = np.percentile(sorted_std_eps, [25, 75])
    slope = (y75 - y25) / (x75 - x25)
    intercept = y25 - slope * x25
    ax2.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, color='red', lw=1.5, linestyle='--')
    ax2.set_title("Normal Q-Q", fontsize=12, fontweight='semibold')
    ax2.set_xlabel("Theoretical Quantiles", fontsize=10)
    ax2.set_ylabel("Standardized Residuals", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # 3. Scale-Location
    ax3 = axes[1, 0]
    sqrt_abs_std_eps = np.sqrt(np.abs(std_eps))
    ax3.scatter(y_hat, sqrt_abs_std_eps, alpha=0.5, color='#9467bd')
    poly_coefs_scale = np.polyfit(y_hat, sqrt_abs_std_eps, 2)
    fit_y_scale = np.polyval(poly_coefs_scale, fit_x)
    ax3.plot(fit_x, fit_y_scale, color='darkblue', lw=2, label='Trend')
    ax3.set_title("Scale-Location (Homoscedasticity check)", fontsize=12, fontweight='semibold')
    ax3.set_xlabel("Fitted Values", fontsize=10)
    ax3.set_ylabel(r"$\sqrt{|Standardized\ Residuals|}$", fontsize=10)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend()
    
    # 4. Cook's Distance
    ax4 = axes[1, 1]
    obs_idx = np.arange(n)
    ax4.stem(obs_idx, cooks_d, markerfmt=' ', linefmt='red', basefmt='gray')
    threshold = 4.0 / n
    ax4.axhline(threshold, color='blue', linestyle='--', lw=1.5, label=f'Threshold (4/n = {threshold:.4f})')
    ax4.set_title("Cook's Distance vs Index", fontsize=12, fontweight='semibold')
    ax4.set_xlabel("Observation Index", fontsize=10)
    ax4.set_ylabel("Cook's Distance", fontsize=10)
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend()
    
    # Annotate extreme points
    top3_cooks_idx = np.argsort(cooks_d)[-3:]
    for idx in top3_cooks_idx:
        ax4.text(idx, cooks_d[idx], f" #{idx}", fontsize=8, color='black', fontweight='bold')
        ax1.text(y_hat[idx], residuals[idx], f" #{idx}", fontsize=8, color='black')
        ax2.text(theoretical_quantiles[np.searchsorted(sorted_std_eps, std_eps[idx])], std_eps[idx], f" #{idx}", fontsize=8, color='black')
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Plots] Saved diagnostic plots to {save_path}")
    else:
        plt.show()

# Main Benchmarking Execution Function
def run_benchmarking(data_path, sample_size=10000, random_state=42):
    print(f"\n==============================================")
    print(f"STARTING COMPREHENSIVE BENCHMARKING ENGINE")
    print(f"==============================================\n")
    
    # 1. Load data
    print(f"[Data Loading] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  - Loaded dataset with shape: {df.shape}")
    
    # 2. Extract Month from Date to capture seasonality
    print("[Feature Engineering] Extracting Month from Date...")
    df['Month'] = pd.to_datetime(df['Date']).dt.month.astype(str)
    
    # 3. Clean Target
    target_col = 'MaxTemp'
    print(f"[Target Definition] Target set to '{target_col}'")
    initial_rows = len(df)
    df = df.dropna(subset=[target_col]).copy()
    print(f"  - Dropped {initial_rows - len(df)} rows with missing target values. Remaining: {len(df)}")
    
    # 4. Drop Leakage columns
    leakage_cols = ['RISK_MM', 'RainTomorrow', 'Date']
    df.drop(columns=[col for col in leakage_cols if col in df.columns], inplace=True)
    
    # 5. Define Feature subsets
    num_cols = ['MinTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 
                'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
                'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm']
    cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'Month']
    
    # Intersection with actual dataframe columns to prevent missing errors
    num_cols = [col for col in num_cols if col in df.columns]
    cat_cols = [col for col in cat_cols if col in df.columns]
    
    # 6. Sample dataset to speed up Coordinate Descent Lasso and preserve performance
    print(f"[Sampling] Sampling {sample_size} rows with seed {random_state} for efficiency...")
    df_sampled = df.sample(n=min(sample_size, len(df)), random_state=random_state).copy()
    print(f"  - Active sample shape: {df_sampled.shape}")
    
    # 7. Train/Test Split (80/20)
    print("[Data Partitioning] Splitting data into 80% Train and 20% Test...")
    n = len(df_sampled)
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    split_point = int(n * 0.8)
    
    train_df = df_sampled.iloc[indices[:split_point]].copy()
    test_df = df_sampled.iloc[indices[split_point:]].copy()
    
    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()
    
    # 8. Stateful Preprocessing using the DataPipeline
    print("[Preprocessing] Executing stateful DataPipeline...")
    pipeline = DataPipeline(strategy='median')
    pipeline.fit(train_df, num_cols, cat_cols)
    
    X_train, column_names = pipeline.transform(train_df)
    X_test, _ = pipeline.transform(test_df)
    
    print(f"  - Preprocessed training matrix shape: {X_train.shape}")
    print(f"  - Preprocessed testing matrix shape: {X_test.shape}")
    
    # List to record metrics for each model
    results = {}
    coefficients = {}
    
    # ==========================================
    # MODEL 1: Baseline OLS
    # ==========================================
    print("\n--- Model 1: Baseline OLS (All Columns) ---")
    beta_ols, sigma2_ols, _, _ = ols_fit(X_train, y_train)
    y_pred_ols = X_test @ beta_ols
    
    results['Baseline OLS'] = {
        'MAE': compute_mae(y_test, y_pred_ols),
        'RMSE': compute_rmse(y_test, y_pred_ols),
        'R2_test': compute_r2(y_test, y_pred_ols)
    }
    coefficients['Baseline OLS'] = beta_ols
    print(f"  MAE: {results['Baseline OLS']['MAE']:.4f} | RMSE: {results['Baseline OLS']['RMSE']:.4f} | R2_test: {results['Baseline OLS']['R2_test']:.4f}")
    
    # ==========================================
    # MODEL 2: Feature-Selected OLS
    # ==========================================
    print("\n--- Model 2: Feature-Selected OLS ---")
    selected_indices = run_feature_selection(X_train, y_train, column_names)
    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]
    
    beta_sel, sigma2_sel, _, _ = ols_fit(X_train_sel, y_train)
    y_pred_sel = X_test_sel @ beta_sel
    
    results['Feature-Selected OLS'] = {
        'MAE': compute_mae(y_test, y_pred_sel),
        'RMSE': compute_rmse(y_test, y_pred_sel),
        'R2_test': compute_r2(y_test, y_pred_sel)
    }
    # Map coefficients back to full column list (0 for omitted)
    full_beta_sel = np.zeros(X_train.shape[1])
    full_beta_sel[selected_indices] = beta_sel
    coefficients['Feature-Selected OLS'] = full_beta_sel
    print(f"  MAE: {results['Feature-Selected OLS']['MAE']:.4f} | RMSE: {results['Feature-Selected OLS']['RMSE']:.4f} | R2_test: {results['Feature-Selected OLS']['R2_test']:.4f}")
    
    # ==========================================
    # MODEL 3: Tuned Ridge Regression
    # ==========================================
    print("\n--- Model 3: Tuned Ridge Regression ---")
    lambdas_ridge = np.logspace(-3, 3, 10)
    print(f"  - Sweeping Ridge lambdas: {list(np.round(lambdas_ridge, 4))}")
    best_lam_ridge, ridge_cv_mses = kfold_cv_tune(X_train, y_train, k=5, lambdas=lambdas_ridge, model_type='ridge')
    print(f"  - Optimal Ridge lambda found: {best_lam_ridge:.4f} (CV MSE: {ridge_cv_mses[best_lam_ridge]:.4f})")
    
    # Train best Ridge
    beta_ridge = ridge_fit(X_train, y_train, best_lam_ridge)
    y_pred_ridge = X_test @ beta_ridge
    
    results['Optimized Ridge'] = {
        'MAE': compute_mae(y_test, y_pred_ridge),
        'RMSE': compute_rmse(y_test, y_pred_ridge),
        'R2_test': compute_r2(y_test, y_pred_ridge)
    }
    coefficients['Optimized Ridge'] = beta_ridge
    print(f"  MAE: {results['Optimized Ridge']['MAE']:.4f} | RMSE: {results['Optimized Ridge']['RMSE']:.4f} | R2_test: {results['Optimized Ridge']['R2_test']:.4f}")
    
    # ==========================================
    # MODEL 4: Tuned Lasso Regression
    # ==========================================
    print("\n--- Model 4: Tuned Lasso Regression ---")
    lambdas_lasso = np.logspace(-4, 1, 10)
    print(f"  - Sweeping Lasso lambdas: {list(np.round(lambdas_lasso, 5))}")
    best_lam_lasso, lasso_cv_mses = kfold_cv_tune(X_train, y_train, k=5, lambdas=lambdas_lasso, model_type='lasso')
    print(f"  - Optimal Lasso lambda found: {best_lam_lasso:.5f} (CV MSE: {lasso_cv_mses[best_lam_lasso]:.4f})")
    
    # Train best Lasso
    beta_lasso = lasso_fit(X_train, y_train, best_lam_lasso, max_iter=1000, tol=1e-4)
    y_pred_lasso = X_test @ beta_lasso
    
    results['Optimized Lasso'] = {
        'MAE': compute_mae(y_test, y_pred_lasso),
        'RMSE': compute_rmse(y_test, y_pred_lasso),
        'R2_test': compute_r2(y_test, y_pred_lasso)
    }
    coefficients['Optimized Lasso'] = beta_lasso
    print(f"  MAE: {results['Optimized Lasso']['MAE']:.4f} | RMSE: {results['Optimized Lasso']['RMSE']:.4f} | R2_test: {results['Optimized Lasso']['R2_test']:.4f}")
    
    # ==========================================
    # MODEL 5: Exact Kernel Ridge Regression
    # ==========================================
    print("\n--- Model 5: Exact Kernel Ridge Regression ---")

    # KRR uses distances; remove explicit intercept if it exists.
    X_train_krr = remove_intercept_column(X_train, column_names)
    X_test_krr = remove_intercept_column(X_test, column_names)


    print(f"  - Exact KRR training subset: {X_train_krr.shape}")

    best_krr, best_krr_params, krr_tuning_df = tune_krr(
        X_train=X_train_krr,
        y_train=y_train,
        val_size=0.2,
        random_state=random_state,
        batch_size=5000
    )

    y_pred_krr = best_krr.predict(X_test_krr, batch_size=5000)

    results['Kernel Ridge'] = {
        'MAE': compute_mae(y_test, y_pred_krr),
        'RMSE': compute_rmse(y_test, y_pred_krr),
        'R2_test': compute_r2(y_test, y_pred_krr)
    }

    print(f"  - Best KRR params: {best_krr_params}")
    print(f"  MAE: {results['Kernel Ridge']['MAE']:.4f} | RMSE: {results['Kernel Ridge']['RMSE']:.4f} | R2_test: {results['Kernel Ridge']['R2_test']:.4f}")


    # ==========================================
    # MODEL 6: Bayesian Linear Regression
    # ==========================================
    print("\n--- Model 6: Bayesian Linear Regression ---")

    best_bayes, best_bayes_params, bayes_tuning_df = tune_bayesian_regression(
        X_train=X_train,
        y_train=y_train,
        alpha_values=None,
        sigma2_values=None,
        val_size=0.2,
        random_state=random_state
    )

    y_pred_bayes, y_std_bayes = best_bayes.predict(X_test, return_std=True)

    results['Bayesian Linear Regression'] = {
        'MAE': compute_mae(y_test, y_pred_bayes),
        'RMSE': compute_rmse(y_test, y_pred_bayes),
        'R2_test': compute_r2(y_test, y_pred_bayes)
    }

    coefficients['Bayesian Linear Regression'] = best_bayes.posterior_mean

    print(f"  - Best Bayesian params: {best_bayes_params}")
    print(f"  MAE: {results['Bayesian Linear Regression']['MAE']:.4f} | RMSE: {results['Bayesian Linear Regression']['RMSE']:.4f} | R2_test: {results['Bayesian Linear Regression']['R2_test']:.4f}")

    # Save tuning outputs for report
    try:
        krr_tuning_df.to_csv("krr_tuning_results.csv", index=False)
        bayes_tuning_df.to_csv("bayesian_tuning_results.csv", index=False)

        lower_95 = y_pred_bayes - 1.96 * y_std_bayes
        upper_95 = y_pred_bayes + 1.96 * y_std_bayes

        bayes_intervals = pd.DataFrame({
            "Actual": y_test[:50],
            "Prediction": y_pred_bayes[:50],
            "Lower_95": lower_95[:50],
            "Upper_95": upper_95[:50]
        })
        bayes_intervals.to_csv("bayesian_prediction_intervals.csv", index=False)

        print("[Save] Saved KRR/Bayesian tuning and interval CSV files.")
    except Exception as e:
        print(f"[Warning] Could not save advanced-model CSV files: {e}")

    # ==========================================
    # PERFORMANCE COMPARISON TABLE
    # ==========================================
    print("\n==============================================")
    print("FINAL TESTING EVALUATION PERFORMANCE MATRIX")
    print("==============================================")
    # Custom markdown formatter to avoid dependency on 'tabulate'
    def df_to_md(df):
        cols = list(df.columns)
        header_line = "| Model | " + " | ".join(cols) + " |"
        separator_line = "| :--- | " + " | ".join([":---:" for _ in cols]) + " |"
        lines_md = [header_line, separator_line]
        for idx, row in df.iterrows():
            row_values = [f"{row[col]:.6f}" if isinstance(row[col], (float, np.floating)) else str(row[col]) for col in cols]
            lines_md.append(f"| {idx} | " + " | ".join(row_values) + " |")
        return "\n".join(lines_md)

    df_metrics = pd.DataFrame(results).T
    print(df_to_md(df_metrics))
    print("==============================================")

    # Save metrics table to CSV for documentation
    df_metrics.to_csv("model_comparison_results.csv")
    print("[Save] Saved metrics table to 'part2/model_comparison_results.csv'")

    # ==========================================
    # COEFFICIENT ANALYSIS
    # ==========================================
    print("\nFeature Coefficients and Sparsity Comparison:")
    df_coefs = pd.DataFrame(index=column_names)
    for model_name, beta in coefficients.items():
        df_coefs[model_name] = beta

    # Calculate non-zero weights count
    sparsity = {name: np.sum(np.abs(beta[1:]) > 1e-4) for name, beta in coefficients.items()}
    print("\nNumber of Active (Non-zero) Features (excluding Intercept):")
    for name, cnt in sparsity.items():
        print(f"  - {name}: {cnt} / {len(column_names)-1}")

    # Top 10 predictive features in best model
    best_model_name = df_metrics['R2_test'].idxmax()
    print(f"\nBest Model identified by R2_test: {best_model_name}")
    print(f"\nTop 10 strongest features (absolute weight) in the Best Model ({best_model_name}):")
    best_beta = coefficients[best_model_name]
    # Skip intercept at index 0 for ranking features
    features_only = column_names[1:]
    weights_only = best_beta[1:]
    sorted_idx = np.argsort(np.abs(weights_only))[::-1]

    for rank in range(min(10, len(features_only))):
        idx = sorted_idx[rank]
        print(f"  {rank+1:2d}. {features_only[idx]:<25}: Weight = {weights_only[idx]:.4f}")

    # ==========================================
    # DIAGNOSTICS PLOTS FOR THE BEST MODEL
    # ==========================================
    print(f"\nGenerating analytical diagnostics for best-performing model: {best_model_name}...")

    if best_model_name == 'Feature-Selected OLS':
        residual_plots(
            X_train_sel,
            y_train,
            beta_sel,
            save_path="best_model_diagnostics.png"
        )

    elif best_model_name in coefficients:
        residual_plots(
            X_train,
            y_train,
            coefficients[best_model_name],
            save_path="best_model_diagnostics.png"
        )

    elif best_model_name == 'Kernel Ridge':
        # Kernel Ridge has no original-space beta coefficients,
        # so use a direct residual-vs-fitted diagnostic.
        train_pred_krr = best_krr.predict(X_train_krr, batch_size=5000)
        residuals_krr = y_train - train_pred_krr

        plt.figure(figsize=(8, 5))
        plt.scatter(train_pred_krr, residuals_krr, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Exact Kernel Ridge: Residuals vs Fitted")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig("best_model_diagnostics.png", dpi=300)
        plt.close()

        print("[Plots] Saved kernel residual plot to best_model_diagnostics.png")

    else:
        print(f"[Warning] No diagnostic plot rule defined for {best_model_name}.")

    print("\n==============================================")
    print("BENCHMARKING AND COMPARISON COMPLETE")
    print("==============================================\n")

if __name__ == '__main__':
    # Ensure run on the weather dataset
    data_file = "part2/data/weatherAUS.csv"
    if os.path.exists(data_file):
        run_benchmarking(data_file, sample_size=10000, random_state=42)
    else:
        print(f"[Error] Dataset file not found at {data_file}. Please check directory structure.")
