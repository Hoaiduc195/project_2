try:
    from matrix_ops import *
except ModuleNotFoundError:
    from part1.matrix_ops import *

import scipy.stats as stats
import random
import math


def ols_fit(X, y):
    X_int = add_intercept(X)
    n = len(X_int)
    p_plus_1 = len(X_int[0])
    p = p_plus_1 - 1
    
    Xt = transpose(X_int)
    XtX = matmul(Xt, X_int)
    XtX_inv = invert(XtX)
    
    y_col = [[yi] for yi in y]
    Xty = matmul(Xt, y_col)
    
    beta_hat_col = matmul(XtX_inv, Xty)
    beta_hat = [b[0] for b in beta_hat_col]
    
    y_hat_col = matmul(X_int, beta_hat_col)
    y_hat = [yhc[0] for yhc in y_hat_col]
    
    rss = sum((y[i] - y_hat[i])**2 for i in range(n))
    sigma2 = rss / (n - p - 1)
    
    return beta_hat, sigma2

def hat_matrix(X):
    X_int = add_intercept(X)
    Xt = transpose(X_int)
    XtX = matmul(Xt, X_int)
    XtX_inv = invert(XtX)
    H = matmul(X_int, matmul(XtX_inv, Xt))
    
    # Check idempotent
    H2 = matmul(H, H)
    diff = sum(abs(H2[i][j] - H[i][j]) for i in range(len(H)) for j in range(len(H[0])))
    if diff > 1e-5:
        print("Cảnh báo: Ma trận Hat không có tính idempotent!")
    return H

def model_metrics(y, y_hat, p):
    n = len(y)
    y_bar = sum(y) / n
    rss = sum((y[i] - y_hat[i])**2 for i in range(n))
    tss = sum((y[i] - y_bar)**2 for i in range(n))
    
    r_squared = 1 - (rss / tss) if tss != 0 else 0
    adj_r_squared = 1 - ((n - 1) / (n - p - 1)) * (1 - r_squared)
    
    f_stat = ((tss - rss) / p) / (rss / (n - p - 1)) if rss != 0 else float('inf')
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
    X_int = add_intercept(X)
    n = len(X_int)
    p = len(X_int[0]) - 1
    
    Xt = transpose(X_int)
    XtX = matmul(Xt, X_int)
    XtX_inv = invert(XtX)
    
    var_beta = scalar_mul(sigma2, XtX_inv)
    se_beta = [math.sqrt(var_beta[i][i]) for i in range(len(var_beta))]
    
    t_stats = [beta_hat[i] / se_beta[i] for i in range(len(beta_hat))]
    p_values = [2 * (1 - stats.t.cdf(abs(t), df=n - p - 1)) for t in t_stats]
    
    t_crit = stats.t.ppf(0.975, df=n - p - 1)
    ci_lower = [beta_hat[i] - t_crit * se_beta[i] for i in range(len(beta_hat))]
    ci_upper = [beta_hat[i] + t_crit * se_beta[i] for i in range(len(beta_hat))]
    
    return {
        'se': se_beta,
        't_stat': t_stats,
        'p_value': p_values,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def vif(X):
    # Prepare matrix without intercept for VIF calculation
    if not isinstance(X[0], list):
        X = [[x] for x in X]
        
    X_vars = []
    has_int = all(row[0] == 1.0 for row in X)
    for row in X:
        if has_int:
            X_vars.append(row[1:])
        else:
            X_vars.append(row[:])
            
    p = len(X_vars[0])
    vif_values = []
    
    for j in range(p):
        X_j = [row[j] for row in X_vars]
        X_others = [[row[k] for k in range(p) if k != j] for row in X_vars]
        
        # Regress X_j on X_others
        X_o_int = add_intercept(X_others)
        Xt = transpose(X_o_int)
        XtX_inv = invert(matmul(Xt, X_o_int))
        y_col = [[x] for x in X_j]
        Xty = matmul(Xt, y_col)
        beta_j = matmul(XtX_inv, Xty)
        
        y_hat_col = matmul(X_o_int, beta_j)
        y_hat_j = [yh[0] for yh in y_hat_col]
        
        xj_mean = sum(X_j) / len(X_j)
        rss_j = sum((X_j[i] - y_hat_j[i])**2 for i in range(len(X_j)))
        tss_j = sum((X_j[i] - xj_mean)**2 for i in range(len(X_j)))
        
        r2_j = 1 - (rss_j / tss_j) if tss_j != 0 else 0
        vif_j = 1 / (1 - r2_j) if r2_j != 1 else float('inf')
        vif_values.append(vif_j)
        
    return vif_values

def gauss_markov_monte_carlo(n_simulations=1000, n=100, p=2, true_beta=None, sigma=1.0):
    if true_beta is None:
        true_beta = [1.0] * (p + 1)
        
    X = [[random.gauss(0, 1) for _ in range(p)] for _ in range(n)]
    X_int = add_intercept(X)
    
    ols_betas = []
    other_betas = []
    
    # C matrix for alternative unbiased estimator
    C_rand = [[random.gauss(0, 1) for _ in range(n)] for _ in range(p + 1)]
    H = hat_matrix(X_int)
    I = eye(n)
    I_minus_H = mat_sub(I, H)
    C = matmul(C_rand, I_minus_H)
    C = scalar_mul(0.1, C)
    
    Xt = transpose(X_int)
    XtX_inv = invert(matmul(Xt, X_int))
    
    for _ in range(n_simulations):
        eps = [random.gauss(0, sigma) for _ in range(n)]
        y_true_col = matmul(X_int, [[b] for b in true_beta])
        y = [y_true_col[i][0] + eps[i] for i in range(n)]
        
        y_col = [[yi] for yi in y]
        b_ols_col = matmul(XtX_inv, matmul(Xt, y_col))
        b_ols = [b[0] for b in b_ols_col]
        ols_betas.append(b_ols)
        
        C_y_col = matmul(C, y_col)
        b_other = [b_ols[i] + C_y_col[i][0] for i in range(len(b_ols))]
        other_betas.append(b_other)
        
    mean_ols = [sum(ols_betas[i][j] for i in range(n_simulations)) / n_simulations for j in range(p + 1)]
    mean_other = [sum(other_betas[i][j] for i in range(n_simulations)) / n_simulations for j in range(p + 1)]
    
    var_ols = [sum((ols_betas[i][j] - mean_ols[j])**2 for i in range(n_simulations)) / n_simulations for j in range(p + 1)]
    var_other = [sum((other_betas[i][j] - mean_other[j])**2 for i in range(n_simulations)) / n_simulations for j in range(p + 1)]
    
    return {
        'true_beta': true_beta,
        'mean_ols': mean_ols,
        'mean_other': mean_other,
        'var_ols': var_ols,
        'var_other': var_other
    }

def ols_predict(X, beta_hat):
    X_int = add_intercept(X)

    beta_col = [[b] for b in beta_hat]

    y_hat_col = matmul(X_int, beta_col)

    y_hat = [row[0] for row in y_hat_col]

    return y_hat