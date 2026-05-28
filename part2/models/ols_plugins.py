from ..model_registry import register_model
from .basic_ols_model import BasicOLSModel
from .selected_ols_model import SelectedOLSModel
from .lib_ols_model import LibOLSModel

@register_model('ols_basic')
class OLSBasicPlugin:
	def __init__(self, **kwargs):
		self.name = 'ols_basic'
		self.model = BasicOLSModel(**kwargs)

	def fit(self, X_train, y_train):
		return self.model.fit(X_train, y_train)

	def predict(self, X_test):
		return self.model.predict(X_test)

@register_model('ols_selected')
class OLSSelectedPlugin:
	def __init__(self, **kwargs):
		self.name = 'ols_selected'
		self.model = SelectedOLSModel(**kwargs)

	def fit(self, X_train, y_train, feature_names):
		return self.model.fit(X_train, y_train, feature_names)

	def predict(self, X_test):
		return self.model.predict(X_test)

@register_model('ols_lib')
class OLSLibPlugin:
	def __init__(self, **kwargs):
		self.name = 'ols_lib'
		self.model = LibOLSModel(**kwargs)

	def fit(self, X_train, y_train):
		return self.model.fit(X_train, y_train)

	def predict(self, X_test):
		return self.model.predict(X_test)
