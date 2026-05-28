from statsmodels import api as sm

class LibOLSModel:
	def __init__(self):
		self.model = None
		self.results = None

	def fit(self, X, y):
		X = sm.add_constant(X)  # Add intercept term
		self.model = sm.OLS(y, X)
		self.results = self.model.fit()

	def predict(self, X_new):
		X_new = sm.add_constant(X_new)  # Add intercept term
		return self.results.predict(X_new)

	def summary(self):
		return self.results.summary()