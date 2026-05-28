import pandas as pd
from part1.ols_implementation import model_metrics
from .eda import EDA
from .preprocessing import Preprocessor
from .model_registry import build_models
from .utils import metrics_summary, test_metrics_summary

class DataPipeline:
	def __init__(self, data_source, target_col, model_names=None, model_params=None):
		self.data_source = data_source
		self.target_col = target_col
		self.preprocessor = Preprocessor()
		self.eda = EDA()
		self.train_data = None
		self.test_data = None
		self.models = build_models(
			model_names=model_names,
			model_params=model_params,
		)

	def extract(self):
		self.df = pd.read_csv(self.data_source)
		self.df = self.df.dropna(subset=[self.target_col]).reset_index(drop=True)
		self.preprocessor.load_df(self.df)
		self.eda.load_df(self.df)
		self.train_data, self.test_data = self.preprocessor.get_processed_data()
		self.X_train = self.train_data.drop(columns=[self.target_col]).values.tolist()
		self.y_train = self.train_data[self.target_col].tolist()
		self.X_test = self.test_data.drop(columns=[self.target_col]).values.tolist()
		self.y_test = self.test_data[self.target_col].tolist()

	def fit(self):
		if self.train_data is None or self.test_data is None:
			raise ValueError("Call extract() before fit().")

		for model in self.models:
			print(f"\n[PIPELINE] Fitting model: {model.name}")
			feature_names = self.train_data.drop(columns=[self.target_col]).columns.tolist()
			if model.name == 'ols_selected':
				model.fit(self.X_train, self.y_train, feature_names)
			else:
				model.fit(self.X_train, self.y_train)
			y_pred = model.predict(self.X_train)
			# metrics = model_metrics(self.y_train, y_pred, p=len(feature_names))
			# metrics_summary(metrics)
		return self

	def evaluate(self):
		if self.test_data is None:
			raise ValueError("Call fit() before evaluate().")
		for model in self.models:
			print(f"\n[PIPELINE] Evaluating model: {model.name}")
			y_pred = model.predict(self.X_test)
			test_metrics_summary(self.y_test, y_pred)
		return self
		
	def run(self):
		self.extract()
		self.fit()
		self.evaluate()

if __name__ == "__main__":
	pipeline = DataPipeline(data_source='part2/weatherAUS.csv', target_col='RISK_MM', model_names=['ols_basic', 'ols_selected', 'ols_lib'])
	pipeline.run()