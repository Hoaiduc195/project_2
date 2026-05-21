from matrix_ops import *
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


def residual_plots(X, y, beta_hat):
    X_int = add_intercept(X)
    n = len(X_int)
    p_plus_1 = len(X_int[0])
    p = p_plus_1 - 1
    
    y_hat_col = matmul(X_int, [[b] for b in beta_hat])
    y_hat = [yh[0] for yh in y_hat_col]
    residuals = [y[i] - y_hat[i] for i in range(n)]
    
    Xt = transpose(X_int)
    XtX_inv = invert(matmul(Xt, X_int))
    H = matmul(X_int, matmul(XtX_inv, Xt))
    leverage = [H[i][i] for i in range(n)]
    
    rss = sum(r**2 for r in residuals)
    sigma_hat = math.sqrt(rss / (n - p - 1))
    
    std_residuals = []
    cooks_d = []
    for i in range(n):
        den = sigma_hat * math.sqrt(1 - leverage[i])
        sr = residuals[i] / den if den != 0 else 0
        std_residuals.append(sr)
        
        cd = (sr**2 * leverage[i]) / (p_plus_1 * (1 - leverage[i])) if (1 - leverage[i]) != 0 else 0
        cooks_d.append(cd)
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Residuals vs Fitted
    axes[0, 0].scatter(y_hat, residuals, alpha=0.6, edgecolors='w', label='Observations')
    axes[0, 0].axhline(0, color='r', linestyle='--', label='Zero Reference')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].legend()
    
    # 2. Normal Q-Q
    stats.probplot(std_residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].get_lines()[0].set_label('Observed Quantiles')
    axes[0, 1].get_lines()[1].set_label('Theoretical Line')
    axes[0, 1].set_xlabel('Theoretical Quantiles')
    axes[0, 1].set_ylabel('Standardized Residuals')
    axes[0, 1].set_title('Normal Q-Q')
    axes[0, 1].legend()
    
    # 3. Scale-Location
    sqrt_abs_std_res = [math.sqrt(abs(sr)) for sr in std_residuals]
    axes[1, 0].scatter(y_hat, sqrt_abs_std_res, alpha=0.6, edgecolors='w', label='Observations')
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel(r'$\sqrt{|Standardized Residuals|}$')
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].legend()
    
    # 4. Residuals vs Leverage
    axes[1, 1].scatter(leverage, std_residuals, alpha=0.6, edgecolors='w', label='Observations')
    axes[1, 1].axhline(0, color='r', linestyle='--', label='Zero Reference')
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Standardized Residuals')
    axes[1, 1].set_title('Residuals vs Leverage')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return cooks_d
