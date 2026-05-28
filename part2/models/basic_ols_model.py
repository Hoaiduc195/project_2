from part1.ols_implementation import (
	ols_fit,
	ols_predict,
)

class BasicOLSModel:
	def __init__(self):
		self.basic_state = None

	def fit(self, X_train, y_train):
		beta_hat, sigma2 = ols_fit(X_train, y_train)
		self.basic_state = {
			'x': X_train,
			'y': y_train,
			'beta_hat': beta_hat,
			'sigma2': sigma2,
		}
		return beta_hat, sigma2

	def predict(self, X_test):
		if self.basic_state is None:
			raise ValueError("Call fit_basic() before predict_basic().")

		y_pred = ols_predict(X_test, self.basic_state['beta_hat'])

		return y_pred
