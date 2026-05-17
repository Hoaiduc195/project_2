import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def add_intercept(X):
    """Thêm cột toàn 1 vào ma trận X nếu chưa có (intercept)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.allclose(X[:, 0], 1):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X

def residual_plots(X, y, beta_hat):
    """
    Vẽ 4 biểu đồ phân tích phần dư:
    1. Residuals vs Fitted
    2. Normal Q-Q
    3. Scale-Location
    4. Residuals vs Leverage
    """
    X_intercept = add_intercept(X)
    n, p_plus_1 = X_intercept.shape
    p = p_plus_1 - 1
    
    y_hat = X_intercept @ beta_hat
    residuals = y - y_hat
    
    # Tính Leverage (H)
    H = X_intercept @ np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T
    leverage = np.diag(H)
    
    # Phần dư chuẩn hóa (Standardized Residuals)
    sigma_hat = np.sqrt(np.sum(residuals**2) / (n - p - 1))
    std_residuals = residuals / (sigma_hat * np.sqrt(1 - leverage))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_hat, residuals, alpha=0.6, edgecolors='w')
    axes[0, 0].axhline(0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # 2. Normal Q-Q
    stats.probplot(std_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q')
    
    # 3. Scale-Location
    axes[1, 0].scatter(y_hat, np.sqrt(np.abs(std_residuals)), alpha=0.6, edgecolors='w')
    # Thêm đường xu hướng đơn giản (lowess/rolling mean could be better but sticking to basic)
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('$\sqrt{|Standardized Residuals|}$')
    axes[1, 0].set_title('Scale-Location')
    
    # 4. Residuals vs Leverage (với Cook's Distance)
    cooks_d = (std_residuals**2 * leverage) / (p_plus_1 * (1 - leverage))
    axes[1, 1].scatter(leverage, std_residuals, alpha=0.6, edgecolors='w')
    axes[1, 1].axhline(0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Standardized Residuals')
    axes[1, 1].set_title('Residuals vs Leverage')
    
    plt.tight_layout()
    plt.show()
    
    return cooks_d
