import unittest
import math
from matrix_ops import add_intercept, matmul, transpose, invert, eye
from ols_implementation import ols_fit, hat_matrix, model_metrics, coef_inference, vif
from ridge_lasso import ridge_fit, lasso_fit
from cross_validation import kfold_cv

class TestPart1(unittest.TestCase):
    
    # --- UNIT TESTS FOR ols_fit ---
    def test_ols_fit_perfect_linear(self):
        # y = 2 * x (perfect linear slope = 2, intercept = 0)
        X = [[1.0], [2.0], [3.0]]
        y = [2.0, 4.0, 6.0]
        beta, sigma2 = ols_fit(X, y)
        self.assertAlmostEqual(beta[0], 0.0, places=5) # intercept
        self.assertAlmostEqual(beta[1], 2.0, places=5) # slope
        self.assertAlmostEqual(sigma2, 0.0, places=5) # zero variance of error
        
    def test_ols_fit_constant(self):
        # y = 5 (constant line slope = 0, intercept = 5)
        X = [[1.0], [2.0], [3.0]]
        y = [5.0, 5.0, 5.0]
        beta, sigma2 = ols_fit(X, y)
        self.assertAlmostEqual(beta[0], 5.0, places=5) # intercept
        self.assertAlmostEqual(beta[1], 0.0, places=5) # slope
        
    # --- UNIT TESTS FOR hat_matrix ---
    def test_hat_matrix_square_full_rank(self):
        # If X is full rank square matrix with intercept, H must be the Identity matrix
        X = [[2.0], [3.0]] # X_int becomes 2x2 invertible matrix
        H = hat_matrix(X)
        I = eye(2)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(H[i][j], I[i][j], places=5)
                
    def test_hat_matrix_trace(self):
        # Trace of Hat matrix must equal p + 1 (degrees of freedom / rank of X_int)
        X = [[1.0], [2.0], [3.0]]
        H = hat_matrix(X)
        trace = sum(H[i][i] for i in range(len(H)))
        self.assertAlmostEqual(trace, 2.0, places=5) # p = 1, so p + 1 = 2
        
    # --- UNIT TESTS FOR model_metrics ---
    def test_model_metrics_perfect_fit(self):
        y = [1.0, 2.0, 3.0]
        y_hat = [1.0, 2.0, 3.0]
        metrics = model_metrics(y, y_hat, p=1)
        self.assertAlmostEqual(metrics['R_squared'], 1.0, places=5)
        self.assertAlmostEqual(metrics['RSS'], 0.0, places=5)
        
    def test_model_metrics_mean_only(self):
        y = [1.0, 2.0, 3.0]
        y_hat = [2.0, 2.0, 2.0] # constant mean prediction
        metrics = model_metrics(y, y_hat, p=1)
        self.assertAlmostEqual(metrics['R_squared'], 0.0, places=5)
        self.assertAlmostEqual(metrics['RSS'], metrics['TSS'], places=5)
        
    # --- UNIT TESTS FOR coef_inference ---
    def test_coef_inference_p_value_range(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [1.1, 2.0, 2.9, 4.1, 5.0]
        beta, sigma2 = ols_fit(X, y)
        inf = coef_inference(X, y, beta, sigma2)
        for p_val in inf['p_value']:
            self.assertTrue(0.0 <= p_val <= 1.0)
            
    def test_coef_inference_standard_errors_positive(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [1.1, 2.0, 2.9, 4.1, 5.0]
        beta, sigma2 = ols_fit(X, y)
        inf = coef_inference(X, y, beta, sigma2)
        for se in inf['se']:
            self.assertTrue(se > 0.0)
            
    # --- UNIT TESTS FOR vif ---
    def test_vif_uncorrelated(self):
        # Two orthogonal zero-mean features (perfectly uncorrelated, no intercept-collinear issues)
        X = [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
        vifs = vif(X)
        for v in vifs:
            self.assertAlmostEqual(v, 1.0, places=5)
            
    def test_vif_perfectly_correlated(self):
        # Perfectly collinear features: X2 = 2 * X1
        X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]
        vifs = vif(X)
        for v in vifs:
            self.assertEqual(v, float('inf'))
            
    # --- UNIT TESTS FOR ridge_fit ---
    def test_ridge_fit_no_regularization(self):
        # With lambda = 0, Ridge should equal OLS
        X = [[1.0], [2.0], [3.0]]
        y = [2.0, 4.0, 6.0]
        beta_ridge = ridge_fit(X, y, lam_values=0.0, plot=False)
        beta_ols, _ = ols_fit(X, y)
        for i in range(len(beta_ols)):
            self.assertAlmostEqual(beta_ridge[i], beta_ols[i], places=5)
            
    def test_ridge_fit_shrinkage(self):
        # With huge lambda, slope should shrink towards 0 (intercept stays unpenalized)
        X = [[1.0], [2.0], [3.0]]
        y = [2.0, 4.0, 6.0]
        beta_ridge = ridge_fit(X, y, lam_values=1e6, plot=False)
        self.assertAlmostEqual(beta_ridge[1], 0.0, places=3) # slope shrunk to 0
        
    # --- UNIT TESTS FOR lasso_fit ---
    def test_lasso_fit_no_regularization(self):
        # With lambda = 0, Lasso should equal OLS (within numerical tolerance of coordinate descent)
        X = [[1.0], [2.0], [3.0]]
        y = [2.0, 4.0, 6.0]
        beta_lasso = lasso_fit(X, y, lam_values=0.0, plot=False)
        beta_ols, _ = ols_fit(X, y)
        for i in range(len(beta_ols)):
            self.assertAlmostEqual(beta_lasso[i], beta_ols[i], places=3)
            
    def test_lasso_fit_sparsity(self):
        # With high lambda, slope must be exactly 0.0 due to L1 penalty
        X = [[1.0], [2.0], [3.0]]
        y = [2.0, 4.0, 6.0]
        beta_lasso = lasso_fit(X, y, lam_values=100.0, plot=False)
        self.assertEqual(beta_lasso[1], 0.0) # perfectly sparse
        
    # --- UNIT TESTS FOR kfold_cv ---
    def test_kfold_cv_output_shape(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        cv_score, fold_scores = kfold_cv(X, y, k=3, model_type='ols', random_state=42)
        self.assertIsInstance(cv_score, float)
        self.assertEqual(len(fold_scores), 3)
        
    def test_kfold_cv_perfect_prediction(self):
        X = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        cv_score, _ = kfold_cv(X, y, k=3, model_type='ols', random_state=42)
        self.assertAlmostEqual(cv_score, 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
