from matrix_ops import *
import random


def kfold_cv(X, y, k=5, random_state=None):
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
        
        X_train_int = add_intercept(X_train)
        Xt = transpose(X_train_int)
        XtX_inv = invert(matmul(Xt, X_train_int))
        y_train_col = [[yi] for yi in y_train]
        beta_col = matmul(XtX_inv, matmul(Xt, y_train_col))
        
        X_test_int = add_intercept(X_test)
        y_pred_col = matmul(X_test_int, beta_col)
        y_pred = [yp[0] for yp in y_pred_col]
        
        mse = sum((y_test[j] - y_pred[j])**2 for j in range(len(y_test))) / len(y_test)
        mse_scores.append(mse)
        
    cv_score = sum(mse_scores) / k
    return cv_score, mse_scores
