from matrix_ops import *
import random


def kfold_cv(X, y, k=5, model_type='ols', lam=0.0, random_state=None):
    n = len(y)
    indices = list(range(n))
    
    if random_state is not None:
        random.seed(random_state)
        random.shuffle(indices)
        
    fold_sizes = [n // k] * k
    for i in range(n % k):
        fold_sizes[i] += 1
        
    current = 0
    folds = []
    for fold_size in fold_sizes:
        folds.append(indices[current:current + fold_size])
        current += fold_size
        
    mse_scores = []
    
    for i in range(k):
        test_indices = folds[i]
        train_indices = [idx for j in range(k) if j != i for idx in folds[j]]
        
        X_train = [X[idx] for idx in train_indices]
        y_train = [y[idx] for idx in train_indices]
        X_test = [X[idx] for idx in test_indices]
        y_test = [y[idx] for idx in test_indices]
        
        if model_type == 'ols':
            X_train_int = add_intercept(X_train)
            Xt = transpose(X_train_int)
            XtX_inv = invert(matmul(Xt, X_train_int))
            y_train_col = [[yi] for yi in y_train]
            beta_col = matmul(XtX_inv, matmul(Xt, y_train_col))
            beta = [b[0] for b in beta_col]
        elif model_type == 'ridge':
            from ridge_lasso import ridge_fit
            beta = ridge_fit(X_train, y_train, lam, plot=False)
        elif model_type == 'lasso':
            from ridge_lasso import lasso_fit
            beta = lasso_fit(X_train, y_train, lam, plot=False)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
        X_test_int = add_intercept(X_test)
        y_pred = [sum(X_test_int[row_idx][col_idx] * beta[col_idx] for col_idx in range(len(beta))) for row_idx in range(len(X_test))]
        
        mse = sum((y_test[j] - y_pred[j])**2 for j in range(len(y_test))) / len(y_test)
        mse_scores.append(mse)
        
    cv_score = sum(mse_scores) / k
    return cv_score, mse_scores

