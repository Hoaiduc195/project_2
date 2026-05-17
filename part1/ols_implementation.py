import numpy as np
from scipy import stats

def add_intercept(X):
    """Thêm cột toàn 1 vào ma trận X nếu chưa có (intercept)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.allclose(X[:, 0], 1):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X

def ols_fit(X, y):
    """
    Tính beta_hat = (X^T X)^{-1} X^T y và sigma2.
    """
    X_intercept = add_intercept(X)
    n, p_plus_1 = X_intercept.shape
    p = p_plus_1 - 1
    
    # Tính beta_hat
    beta_hat = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y
    
    # Tính sigma2
    y_hat = X_intercept @ beta_hat
    rss = np.sum((y - y_hat) ** 2)
    sigma2 = rss / (n - p - 1)
    
    return beta_hat, sigma2

def hat_matrix(X):
    """
    Tính ma trận chiếu H = X(X^T X)^{-1} X^T và kiểm tra tính idempotent.
    """
    X_intercept = add_intercept(X)
    H = X_intercept @ np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T
    
    # Kiểm tra tính idempotent (H^2 = H)
    is_idempotent = np.allclose(H @ H, H)
    if not is_idempotent:
        print("Cảnh báo: Ma trận Hat không có tính idempotent!")
        
    return H

def model_metrics(y, y_hat, p):
    """
    Tính RSS, TSS, R^2, R^2 hiệu chỉnh, kiểm định F.
    """
    n = len(y)
    
    rss = np.sum((y - y_hat) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    
    r_squared = 1 - (rss / tss)
    adj_r_squared = 1 - ((n - 1) / (n - p - 1)) * (1 - r_squared)
    
    # Kiểm định F cho mô hình tổng thể
    f_stat = ((tss - rss) / p) / (rss / (n - p - 1))
    p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)
    
    return {
        'RSS': rss,
        'TSS': tss,
        'R_squared': r_squared,
        'Adj_R_squared': adj_r_squared,
        'F_stat': f_stat,
        'F_p_value': p_value
    }

def coef_inference(X, y, beta_hat, sigma2):
    """
    Tính standard errors, t-statistics, p-values và khoảng tin cậy 95% cho các hệ số.
    """
    X_intercept = add_intercept(X)
    n, p_plus_1 = X_intercept.shape
    p = p_plus_1 - 1
    
    var_beta = sigma2 * np.linalg.inv(X_intercept.T @ X_intercept)
    se_beta = np.sqrt(np.diag(var_beta))
    
    t_stats = beta_hat / se_beta
    # P-value hai phía
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - p - 1))
    
    # Khoảng tin cậy 95%
    t_crit = stats.t.ppf(0.975, df=n - p - 1)
    ci_lower = beta_hat - t_crit * se_beta
    ci_upper = beta_hat + t_crit * se_beta
    
    return {
        'se': se_beta,
        't_stat': t_stats,
        'p_value': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def vif(X):
    """
    Tính VIF (Variance Inflation Factor) cho từng biến (không tính intercept).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Đảm bảo không tính VIF cho cột intercept
    if np.allclose(X[:, 0], 1):
        X_vars = X[:, 1:]
    else:
        X_vars = X
        
    p = X_vars.shape[1]
    vif_values = []
    
    for j in range(p):
        X_j = X_vars[:, j]
        # Sử dụng các biến còn lại làm predictors
        mask = np.ones(p, dtype=bool)
        mask[j] = False
        X_others = X_vars[:, mask]
        
        # OLS hồi quy X_j theo X_others
        X_others_intercept = add_intercept(X_others)
        beta_j = np.linalg.inv(X_others_intercept.T @ X_others_intercept) @ X_others_intercept.T @ X_j
        y_hat_j = X_others_intercept @ beta_j
        
        rss_j = np.sum((X_j - y_hat_j)**2)
        tss_j = np.sum((X_j - np.mean(X_j))**2)
        
        r2_j = 1 - (rss_j / tss_j) if tss_j != 0 else 0
        vif_j = 1 / (1 - r2_j) if r2_j != 1 else np.inf
        vif_values.append(vif_j)
        
    return np.array(vif_values)

def gauss_markov_monte_carlo(n_simulations=1000, n=100, p=2, true_beta=None, sigma=1.0):
    """
    Mô phỏng Monte Carlo để kiểm chứng định lý Gauss-Markov:
    E[beta_hat] = beta và OLS có phương sai nhỏ nhất.
    """
    if true_beta is None:
        true_beta = np.ones(p + 1)
        
    X = np.random.randn(n, p)
    X_intercept = add_intercept(X)
    
    ols_betas = []
    other_betas = []
    
    # Tạo ma trận C để xây dựng một ước lượng tuyến tính không chệch khác:
    # beta_tilde = beta_hat + C y  (với điều kiện C X = 0)
    C_rand = np.random.randn(p + 1, n)
    H = hat_matrix(X_intercept)
    # Chiếu C_rand lên phần bù vuông góc của X (I - H)
    C = C_rand @ (np.eye(n) - H)
    C = C * 0.1 # Thu nhỏ tỉ lệ C
    
    for _ in range(n_simulations):
        eps = np.random.normal(0, sigma, n)
        y = X_intercept @ true_beta + eps
        
        # Ước lượng OLS
        b_ols = np.linalg.inv(X_intercept.T @ X_intercept) @ X_intercept.T @ y
        ols_betas.append(b_ols)
        
        # Ước lượng không chệch khác
        b_other = b_ols + C @ y
        other_betas.append(b_other)
        
    ols_betas = np.array(ols_betas)
    other_betas = np.array(other_betas)
    
    return {
        'true_beta': true_beta,
        'mean_ols': np.mean(ols_betas, axis=0),
        'mean_other': np.mean(other_betas, axis=0),
        'var_ols': np.var(ols_betas, axis=0),
        'var_other': np.var(other_betas, axis=0)
    }
