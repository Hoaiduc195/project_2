import numpy as np

def add_intercept(X):
    """Thêm cột toàn 1 vào ma trận X nếu chưa có (intercept)."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if not np.allclose(X[:, 0], 1):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
    return X

def kfold_cv(X, y, k=5, random_state=None):
    """
    Cài đặt k-fold cross-validation, tính CV score (Mean Squared Error).
    Trả về điểm số CV trung bình và danh sách điểm của từng fold.
    """
    n = len(y)
    indices = np.arange(n)
    
    if random_state is not None:
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1
    
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
        
    mse_scores = []
    
    for i in range(k):
        test_indices = folds[i]
        train_indices = np.hstack([folds[j] for j in range(k) if j != i])
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Fit OLS trên tập train
        X_train_intercept = add_intercept(X_train)
        beta_hat = np.linalg.inv(X_train_intercept.T @ X_train_intercept) @ X_train_intercept.T @ y_train
        
        # Dự đoán trên tập test
        X_test_intercept = add_intercept(X_test)
        y_pred = X_test_intercept @ beta_hat
        
        mse = np.mean((y_test - y_pred)**2)
        mse_scores.append(mse)
        
    cv_score = np.mean(mse_scores)
    return cv_score, mse_scores
