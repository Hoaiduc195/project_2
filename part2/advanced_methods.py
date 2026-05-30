import numpy as np
import pandas as pd
from utils import compute_mae, compute_rmse, compute_r2
# ============================================================
# Advanced Methods: Exact Kernel Ridge and Bayesian Regression
# ============================================================

class KernelRidgeRegression:
    """
    Exact Kernel Ridge Regression from scratch.

    Training:
        alpha = (K + lambda I)^(-1) y

    Prediction:
        y_hat = K_test @ alpha

    Notes:
        This model builds a full n x n Gram matrix, so use a training subset
        for large datasets.
    """

    def __init__(self, kernel="rbf", lambda_reg=1.0, sigma=1.0, degree=2, coef0=1.0):
        self.kernel = kernel
        self.lambda_reg = lambda_reg
        self.sigma = sigma
        self.degree = degree
        self.coef0 = coef0
        self.X_train = None
        self.alpha = None

    def _linear_kernel(self, X1, X2):
        return np.matmul(X1, np.transpose(X2))

    def _polynomial_kernel(self, X1, X2):
        return (np.matmul(X1, np.transpose(X2)) + self.coef0) ** self.degree

    def _rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist_sq = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-dist_sq / (2.0 * self.sigma ** 2))

    def _compute_kernel(self, X1, X2):
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)

        if self.kernel == "linear":
            return self._linear_kernel(X1, X2)
        if self.kernel == "polynomial":
            return self._polynomial_kernel(X1, X2)
        if self.kernel == "rbf":
            return self._rbf_kernel(X1, X2)

        raise ValueError("kernel must be one of: 'linear', 'polynomial', 'rbf'")

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_train = X
        n_samples = X.shape[0]

        print(f"[Exact KRR] Building Gram matrix: {n_samples} x {n_samples}")
        K = self._compute_kernel(X, X)

        A = K + self.lambda_reg * np.eye(n_samples)

        print("[Exact KRR] Solving (K + lambda I) alpha = y")
        self.alpha = np.linalg.solve(A, y)

        return self

    def predict(self, X, batch_size=5000):
        if self.alpha is None:
            raise ValueError("Model has not been fitted yet.")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        preds = []

        for start in range(0, X.shape[0], batch_size):
            end = start + batch_size
            X_batch = X[start:end]
            K_batch = self._compute_kernel(X_batch, self.X_train)
            preds.append(K_batch @ self.alpha)

        return np.concatenate(preds)


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with Gaussian prior and Gaussian likelihood.

    Prior:
        beta ~ N(0, alpha^-1 I)

    Likelihood:
        y | X, beta ~ N(X beta, sigma2 I)

    Posterior:
        beta | X, y ~ N(m_n, S_n)
    """

    def __init__(self, alpha=1.0, sigma2=1.0, fit_intercept=False):
        self.alpha = alpha
        self.sigma2 = sigma2
        self.fit_intercept = fit_intercept
        self.posterior_mean = None
        self.posterior_cov = None

    def _prepare_X(self, X):
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])

        return X

    def fit(self, X, y):
        X = self._prepare_X(X)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_features = X.shape[1]

        S0_inv = self.alpha * np.eye(n_features)

        # If X already contains an intercept column, do not strongly shrink it.
        if np.std(X[:, 0]) < 1e-12:
            S0_inv[0, 0] = 1e-8

        precision = S0_inv + (1.0 / self.sigma2) * (X.T @ X)

        try:
            self.posterior_cov = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            self.posterior_cov = np.linalg.pinv(precision)

        self.posterior_mean = self.posterior_cov @ ((1.0 / self.sigma2) * X.T @ y)

        return self

    def predict(self, X, return_std=False):
        if self.posterior_mean is None:
            raise ValueError("Model has not been fitted yet.")

        X = self._prepare_X(X)
        mean = X @ self.posterior_mean

        if not return_std:
            return mean

        pred_var = self.sigma2 + np.sum((X @ self.posterior_cov) * X, axis=1)
        pred_std = np.sqrt(np.maximum(pred_var, 0.0))

        return mean, pred_std


def remove_intercept_column(X, column_names=None):
    """
    KRR with an RBF kernel should not use the explicit Intercept column
    because it is constant and does not help the distance calculation.
    """
    X = np.asarray(X, dtype=np.float64)

    if column_names is not None and len(column_names) > 0:
        if str(column_names[0]).lower() == "intercept":
            return X[:, 1:]

    if X.shape[1] > 0 and np.std(X[:, 0]) < 1e-12:
        return X[:, 1:]

    return X




def validation_split(X, y, val_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    X = np.asarray(X)
    y = np.asarray(y)

    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)

    val_count = int(n * val_size)

    val_idx = idx[:val_count]
    train_idx = idx[val_count:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def tune_krr(
    X_train,
    y_train,
    lambda_values=None,
    sigma_values=None,
    val_size=0.2,
    random_state=42,
    batch_size=5000
):
    if lambda_values is None:
        lambda_values = [0.01, 0.1, 1.0, 10.0]

    if sigma_values is None:
        sigma_values = [0.5, 1.0, 2.0, 5.0]

    X_subtrain, X_val, y_subtrain, y_val = validation_split(
        X_train,
        y_train,
        val_size=val_size,
        random_state=random_state
    )

    best_rmse = float("inf")
    best_params = None
    best_model = None
    rows = []

    for lam in lambda_values:
        for sigma in sigma_values:
            print("=" * 60)
            print(f"[KRR Tuning] lambda={lam}, sigma={sigma}")

            model = KernelRidgeRegression(
                kernel="rbf",
                lambda_reg=lam,
                sigma=sigma
            )

            model.fit(X_subtrain, y_subtrain)

            pred = model.predict(X_val, batch_size=batch_size)

            mae = compute_mae(y_val, pred)
            rmse = compute_rmse(y_val, pred)
            r2 = compute_r2(y_val, pred)

            print(f"MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

            rows.append({
                "lambda_reg": lam,
                "sigma": sigma,
                "MAE": mae,
                "RMSE": rmse,
                "R2_test": r2
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    "lambda_reg": lam,
                    "sigma": sigma
                }
                best_model = model

    return best_model, best_params, pd.DataFrame(rows)


def tune_bayesian_regression(
    X_train,
    y_train,
    alpha_values=None,
    sigma2_values=None,
    val_size=0.2,
    random_state=42
):
    if alpha_values is None:
        alpha_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    if sigma2_values is None:
        y_var = np.var(y_train)
        sigma2_values = [0.05 * y_var, 0.1 * y_var, 0.5 * y_var, 1.0 * y_var]

    X_subtrain, X_val, y_subtrain, y_val = validation_split(
        X_train,
        y_train,
        val_size=val_size,
        random_state=random_state
    )

    best_rmse = float("inf")
    best_params = None
    best_model = None
    rows = []

    for alpha in alpha_values:
        for sigma2 in sigma2_values:
            print("=" * 60)
            print(f"[Bayes Tuning] alpha={alpha}, sigma2={sigma2:.6f}")

            model = BayesianLinearRegression(
                alpha=alpha,
                sigma2=sigma2,
                fit_intercept=False
            )

            model.fit(X_subtrain, y_subtrain)

            pred = model.predict(X_val)

            mae = compute_mae(y_val, pred)
            rmse = compute_rmse(y_val, pred)
            r2 = compute_r2(y_val, pred)

            print(f"MAE={mae:.4f} | RMSE={rmse:.4f} | R2={r2:.4f}")

            rows.append({
                "alpha": alpha,
                "sigma2": sigma2,
                "MAE": mae,
                "RMSE": rmse,
                "R2_test": r2
            })

            if rmse < best_rmse:
                best_rmse = rmse
                best_params = {
                    "alpha": alpha,
                    "sigma2": sigma2
                }
                best_model = model

    return best_model, best_params, pd.DataFrame(rows)
