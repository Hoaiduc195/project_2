import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from preprocessing import Preprocessor
from eda import EDA
from utils import calculate_vif

class DataPipeline:
	def __init__(self, data_source):
		self.data_source = data_source
		self.target_col = 'RISK_MM'
		self.preprocessor = Preprocessor()
		self.eda = EDA()

	def extract(self):
		# Code to extract data from the source
		self.df = pd.read_csv(self.data_source)
		self.df = self.df.dropna(subset=[self.target_col]).reset_index(drop=True)

	def data_info(self, df):
		print("Data Shape:", df.shape)
		print("\nMissing Values:\n", df.isnull().sum())
		print("\nData Types:\n", df.dtypes)
		print("\nStatistical Summary:\n", df.describe())
		print("\nNumeric Columns:\n", df.select_dtypes(include=[np.number]).columns)
		print("\nCategorical Columns:\n", df.select_dtypes(include=['object']).columns)

	def run(self):
		'''
		TODO:
		'''
		self.extract()
		self.preprocessor.load_df(self.df)
		self.eda.load_df(self.df)
		self.eda.summary_statistics()
		train_data, test_data = self.preprocessor.get_processed_data()
		print("\nTrain Data Info:")
		self.data_info(train_data)
		print("\nTest Data Info:")
		self.data_info(test_data)
		run_ols_models(train_data, test_data)

# def run_ols_models(train_data, test_data):
# 	print("\n" + "="*60)
# 	print("Model 1: Base OLS (ALL FEATURES)")
# 	print("="*60)
	
# 	y_train = train_data[data_pipeline.target_col]
# 	X_train = train_data.drop(columns=[data_pipeline.target_col])
# 	X_train_sm = sm.add_constant(X_train)
# 	ols_basic = sm.OLS(y_train, X_train_sm).fit()
	
# 	print("\n" + "="*60)
# 	print("Model 2: OLS Variable Selection (Dropping VIF > 10 and P-Value > 0.05)")
# 	print("="*60)
	
# 	X_selected = X_train.copy()
	
# 	# VIF check
# 	while True:
# 		vif_df = calculate_vif(X_selected)
# 		max_vif = vif_df['VIF'].max()
# 		if max_vif > 10:
# 			feature_to_drop = vif_df.loc[vif_df['VIF'] == max_vif, 'Feature'].iloc[0]
# 			X_selected = X_selected.drop(columns=[feature_to_drop])
# 		else:
# 			break
			
# 	# p-value check
# 	while True:
# 		X_selected_sm = sm.add_constant(X_selected)
# 		ols_step = sm.OLS(y_train, X_selected_sm).fit()
# 		p_values = ols_step.pvalues.drop('const', errors='ignore') 
# 		max_p_value = p_values.max()
		
# 		if max_p_value > 0.05:
# 			feature_to_drop = p_values.idxmax()
# 			X_selected = X_selected.drop(columns=[feature_to_drop])
# 		else:
# 			break
			
# 	X_final_sm = sm.add_constant(X_selected)
# 	ols_final = sm.OLS(y_train, X_final_sm).fit()
# 	print(f"Final Model Features: {len(X_selected.columns)} variables.")

# 	# Test evaluation
# 	print("\n" + "="*60)
# 	print("Test Evaluation")
# 	print("="*60)
	
# 	y_test = test_data[data_pipeline.target_col]
	
# 	X_test = test_data[X_selected.columns]
	
# 	X_test_sm = sm.add_constant(X_test, has_constant='add')
	
# 	y_pred = ols_final.predict(X_test_sm)
	
# 	mae = mean_absolute_error(y_test, y_pred)
# 	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# 	r2 = r2_score(y_test, y_pred)
	
# 	print(f"1. MAE (Mean Absolute Error) : {mae:.4f} mm")
# 	print(f"2. RMSE (Root Mean Squared Error): {rmse:.4f} mm")
# 	print(f"3. R-squared (R²)              : {r2:.4f}")
def run_ols_models(train_data, test_data):
	import pandas as pd
	import numpy as np
	import statsmodels.api as sm
	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

	print("\n" + "="*60)
	print("Model 1: Base OLS (ALL FEATURES)")
	print("="*60)

	# =========================
	# TRAIN BASE MODEL
	# =========================
	y_train = train_data[data_pipeline.target_col]

	X_train_full = train_data.drop(columns=[data_pipeline.target_col])

	X_train_full_sm = sm.add_constant(X_train_full)

	ols_basic = sm.OLS(y_train, X_train_full_sm).fit()

	print(ols_basic.summary())

	# =========================
	# MODEL 2: VARIABLE SELECTION
	# =========================
	print("\n" + "="*60)
	print("Model 2: OLS Variable Selection")
	print("(Dropping VIF > 10 and P-Value > 0.05)")
	print("="*60)

	X_selected = X_train_full.copy()

	# -------------------------
	# VIF Selection
	# -------------------------
	while True:
		vif_df = calculate_vif(X_selected)

		max_vif = vif_df['VIF'].max()

		if max_vif > 10:
			feature_to_drop = vif_df.loc[
				vif_df['VIF'] == max_vif,
				'Feature'
			].iloc[0]

			print(f"[VIF REMOVE] {feature_to_drop} (VIF={max_vif:.2f})")

			X_selected = X_selected.drop(columns=[feature_to_drop])

		else:
			break

	# -------------------------
	# P-VALUE Selection
	# -------------------------
	while True:
		X_selected_sm = sm.add_constant(X_selected)

		ols_step = sm.OLS(y_train, X_selected_sm).fit()

		p_values = ols_step.pvalues.drop('const', errors='ignore')

		max_p_value = p_values.max()

		if max_p_value > 0.05:
			feature_to_drop = p_values.idxmax()

			print(f"[P-VALUE REMOVE] {feature_to_drop} (p={max_p_value:.4f})")

			X_selected = X_selected.drop(columns=[feature_to_drop])

		else:
			break

	# Final selected model
	X_final_sm = sm.add_constant(X_selected)

	ols_final = sm.OLS(y_train, X_final_sm).fit()

	print("\nFinal Selected Features:")
	print(list(X_selected.columns))

	print(f"\nTotal Features: {len(X_selected.columns)}")

	print(ols_final.summary())

	# ==========================================================
	# TEST EVALUATION
	# ==========================================================
	print("\n" + "="*60)
	print("TEST SET EVALUATION")
	print("="*60)

	y_test = test_data[data_pipeline.target_col]

	# ==========================================================
	# EVALUATE MODEL 1 (FULL FEATURES)
	# ==========================================================
	X_test_full = test_data[X_train_full.columns]

	X_test_full_sm = sm.add_constant(
		X_test_full,
		has_constant='add'
	)

	y_pred_basic = ols_basic.predict(X_test_full_sm)

	mae_basic = mean_absolute_error(y_test, y_pred_basic)
	rmse_basic = np.sqrt(mean_squared_error(y_test, y_pred_basic))
	r2_basic = r2_score(y_test, y_pred_basic)

	# ==========================================================
	# EVALUATE MODEL 2 (SELECTED FEATURES)
	# ==========================================================
	X_test_selected = test_data[X_selected.columns]

	X_test_selected_sm = sm.add_constant(
		X_test_selected,
		has_constant='add'
	)

	y_pred_final = ols_final.predict(X_test_selected_sm)

	mae_final = mean_absolute_error(y_test, y_pred_final)
	rmse_final = np.sqrt(mean_squared_error(y_test, y_pred_final))
	r2_final = r2_score(y_test, y_pred_final)

	# ==========================================================
	# COMPARISON REPORT
	# ==========================================================
	report_df = pd.DataFrame({
		'Model': [
			'OLS Full Features',
			'OLS Selected Features'
		],
		'Num Features': [
			len(X_train_full.columns),
			len(X_selected.columns)
		],
		'MAE': [
			mae_basic,
			mae_final
		],
		'RMSE': [
			rmse_basic,
			rmse_final
		],
		'R2': [
			r2_basic,
			r2_final
		]
	})

	print("\nMODEL COMPARISON")
	print("-"*60)

	print(report_df.round(4))

	return {
		"ols_basic": ols_basic,
		"ols_final": ols_final,
		"report": report_df
	}

if __name__ == "__main__":
	data_pipeline = DataPipeline(data_source='part2/weatherAUS.csv') 
	data_pipeline.run()