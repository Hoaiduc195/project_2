import numpy as np
import matplotlib.pyplot as plt

def vif(X):
    """
    Calculate the Variance Inflation Factor (VIF) for each feature in X.
    
    Parameters:
    -----------
    X : numpy.ndarray of shape (n, p) or (n, p+1)
        The design matrix. Constant columns (intercepts) are automatically 
        identified and excluded from the VIF calculations.
        
    Returns:
    --------
    vifs : numpy.ndarray of shape (p,)
        Variance Inflation Factor for each column of X. 
        Constant columns will have np.nan.
    """
    n, p = X.shape
    
    # Identify non-constant columns
    non_const_indices = []
    const_indices = []
    for j in range(p):
        if np.std(X[:, j]) < 1e-9:
            const_indices.append(j)
        else:
            non_const_indices.append(j)
            
    vifs = np.zeros(p)
    for j in const_indices:
        vifs[j] = np.nan
        
    for j in non_const_indices:
        # Define the target sub-response and sub-features
        y_sub = X[:, j]
        other_indices = [idx for idx in non_const_indices if idx != j]
        X_sub_features = X[:, other_indices]
        
        # Add a constant intercept column for the sub-regression
        X_sub = np.column_stack((np.ones(n), X_sub_features))
        
        try:
            # Solve OLS: (X_sub^T * X_sub) * beta = X_sub^T * y_sub
            beta_sub = np.linalg.solve(X_sub.T @ X_sub, X_sub.T @ y_sub)
            y_hat_sub = X_sub @ beta_sub
            
            # Compute localized R^2
            tss = np.sum((y_sub - np.mean(y_sub))**2)
            rss = np.sum((y_sub - y_hat_sub)**2)
            
            if tss < 1e-9:
                r2 = 0.0
            else:
                r2 = 1.0 - (rss / tss)
                
            # Compute VIF
            if r2 >= 1.0 - 1e-9:
                vifs[j] = float('inf')
            else:
                vifs[j] = 1.0 / (1.0 - r2)
        except np.linalg.LinAlgError:
            # Matrix is singular (perfect multicollinearity)
            vifs[j] = float('inf')
            
    return vifs

def ridge_fit(X, y, lam):
    """
    Fit Ridge Regression from scratch using the closed-form Normal Equations.
    
    Parameters:
    -----------
    X : numpy.ndarray of shape (n, p)
        The design matrix. Assumed to have a constant column at index 0.
    y : numpy.ndarray of shape (n,)
        The target values.
    lam : float
        L2 regularization parameter (lambda >= 0).
        
    Returns:
    --------
    beta : numpy.ndarray of shape (p,)
        Estimated regression coefficients.
    """
    n, p = X.shape
    
    # Exclude intercept beta_0 from L2 regularization
    I = np.identity(p)
    I[0, 0] = 0.0
    
    # Solve the system (X^T * X + lambda * I) * beta = X^T * y
    A = X.T @ X + lam * I
    b = X.T @ y
    
    beta = np.linalg.solve(A, b)
    return beta

def lasso_fit(X, y, lam, max_iter=1000, tol=1e-4):
    """
    Fit Lasso Regression from scratch using Coordinate Descent.
    
    Parameters:
    -----------
    X : numpy.ndarray of shape (n, p)
        The design matrix. Assumed to have a constant column (intercept) at index 0.
    y : numpy.ndarray of shape (n,)
        The target values.
    lam : float
        L1 regularization parameter (lambda >= 0).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
        
    Returns:
    --------
    beta : numpy.ndarray of shape (p,)
        Estimated regression coefficients.
    """
    n, p = X.shape
    beta = np.zeros(p)
    y_hat = np.zeros(n) # Since beta is initialized to zeros
    
    for iteration in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            sum_sq_Xj = np.sum(X[:, j]**2)
            if sum_sq_Xj == 0:
                beta[j] = 0.0
                continue
            
            # Compute rho_j: sum_i X_ij * (y_i - sum_{k != j} X_ik * beta_k)
            # Efficiently: np.dot(X[:, j], y - y_hat) + sum_sq_Xj * beta[j]
            rho_j = np.dot(X[:, j], y - y_hat) + sum_sq_Xj * beta[j]
            
            beta_j_old = beta[j]
            if j == 0:
                # Intercept is not penalized
                beta[j] = rho_j / sum_sq_Xj
            else:
                # Features are penalized with L1
                lambda_n = lam * n
                if rho_j < -lambda_n:
                    beta[j] = (rho_j + lambda_n) / sum_sq_Xj
                elif rho_j > lambda_n:
                    beta[j] = (rho_j - lambda_n) / sum_sq_Xj
                else:
                    beta[j] = 0.0
            
            # Update predictions
            if beta[j] != beta_j_old:
                y_hat += X[:, j] * (beta[j] - beta_j_old)
                
        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            break
            
    return beta

def plot_ridge_trace(X, y, lambdas=None, feature_names=None, save_path=None):
    """
    Plot the Ridge regression coefficient trace across varying lambdas.
    
    Parameters:
    -----------
    X : numpy.ndarray of shape (n, p)
        The design matrix with constant intercept in column 0.
    y : numpy.ndarray of shape (n,)
        The target values.
    lambdas : array-like or None
        List of regularization parameters to scan.
    feature_names : list of str or None
        Names of features to display in the legend.
    save_path : str or None
        File path to save the generated plot. If None, it will be displayed.
    """
    if lambdas is None:
        lambdas = np.logspace(-4, 4, 100)
    
    coefs = []
    for lam in lambdas:
        beta = ridge_fit(X, y, lam)
        coefs.append(beta[1:]) # Skip intercept
    coefs = np.array(coefs)
    
    plt.figure(figsize=(10, 6))
    num_features = coefs.shape[1]
    
    for i in range(num_features):
        name = feature_names[i] if (feature_names is not None and i < len(feature_names)) else f'Feature {i+1}'
        plt.plot(lambdas, coefs[:, i], label=name)
        
    plt.xscale('log')
    plt.xlabel('Regularization parameter $\\lambda$')
    plt.ylabel('Coefficients $\\beta$')
    plt.title('Ridge Regression Coefficient Trace')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
