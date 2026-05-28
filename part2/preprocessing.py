import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import category_encoders as ce

class Preprocessor:
	def __init__(self):
		self.num_imputer = IterativeImputer(random_state=42, max_iter=50)
		self.cat_imputer = SimpleImputer(strategy='most_frequent')
		self.cat_cols_to_impute = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
		self.cat_cols_for_encoding = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
		self.target_col = 'RISK_MM'
		self.bin_encoder = ce.BinaryEncoder(cols=self.cat_cols_for_encoding)
		self.target_encoder = ce.TargetEncoder(
			cols=self.cat_cols_for_encoding,
			smoothing=10 
		)
	
	def load_df(self, df):
		self.df = df
		self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.to_list()
		self.numeric_cols.remove(self.target_col)
		self.categorical_cols = self.df.select_dtypes(include=['object']).columns.to_list()

	def _numeric_missing_imputation(self, df, is_train=True):
		if is_train:
			return pd.DataFrame(self.num_imputer.fit_transform(df), columns=df.columns, index=df.index)
		else:
			return pd.DataFrame(self.num_imputer.transform(df), columns=df.columns, index=df.index)

	def _categorical_missing_imputation(self, df, is_train=True):
		if is_train:
			df[self.cat_cols_to_impute] = self.cat_imputer.fit_transform(df[self.cat_cols_to_impute])
		else:
			df[self.cat_cols_to_impute] = self.cat_imputer.transform(df[self.cat_cols_to_impute])
		return df
	
	def _numeric_outlier_detection(self, df):
		iso = IsolationForest(contamination=0.05, random_state=42)
		preds = iso.fit_predict(df[self.numeric_cols])
		outliers = set(df.index[preds == -1])
		return df.loc[~df.index.isin(outliers)]

	def _categorical_encoding(self, df, is_train=True):
		# 1. Date: Extract month from Date
		df['Date'] = pd.to_datetime(df['Date'])
		df['Month'] = df['Date'].dt.month
		# Cyclical encoding for Month
		df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
		df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

		# 3. Binary Encoding for 'RainToday' and 'RainTomorrow'
		df['RainToday_Num'] = df['RainToday'].map({'Yes': 1, 'No': 0})
		# df['RainTomorrow_Num'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

		# --- Advanced Encoding ---

		# 1. Target Encoding
		# 2. Binary Encoding for wind directions and Location
		# Wind directions have 16 values -> Binary will create 5 columns for each variable (instead of 16 One-hot columns)
		if is_train:
			# df_target_encoded = self.target_encoder.fit_transform(df, df['RainTomorrow_Num'])
			# df_bin_encoded = self.bin_encoder.fit_transform(df[self.bin_cols])
			# df = pd.concat([df, df_target_encoded, df_bin_encoded], axis=1)
			df = self.target_encoder.fit_transform(df, df[self.target_col])
		else:
			# df_target_encoded = self.target_encoder.transform(df)
			# df_bin_encoded = self.bin_encoder.transform(df[self.bin_cols])
			# df = pd.concat([df, df_target_encoded, df_bin_encoded], axis=1)
			df = self.target_encoder.transform(df)

		# Drop original categorical columns after encoding
		cols_to_drop = ['Date', 'Month', 'RainToday', 'RainTomorrow']
		df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

		return df

	def fit_transform(self, df):
		df_num = df[self.numeric_cols].copy()
		df_cat = df[self.categorical_cols].copy()
		target = df[self.target_col].copy()

		# 1. Imputation
		df_num = self._numeric_missing_imputation(df_num, is_train=True)
		df_cat = self._categorical_missing_imputation(df_cat, is_train=True)

		# Combine numeric and categorical dataframes for outlier detection
		df_combined = pd.concat([df_num, df_cat, target], axis=1)

		# 2. Outlier Detection
		df_combined = self._numeric_outlier_detection(df_combined)

		# 3. Encoding
		df_combined = self._categorical_encoding(df_combined, is_train=True)
		return df_combined

	def transform(self, df):
		df_num = df[self.numeric_cols].copy()
		df_cat = df[self.categorical_cols].copy()
		target = df[self.target_col].copy()

		# 1. Imputation using the fitted imputers
		df_num = self._numeric_missing_imputation(df_num, is_train=False)
		df_cat = self._categorical_missing_imputation(df_cat, is_train=False)

		df_combined = pd.concat([df_num, df_cat, target], axis=1)
		
		# Encoding using the fitted encoders
		df_combined = self._categorical_encoding(df_combined, is_train=False)
		return df_combined

	def get_processed_data(self):
		# Split Train / Test
		df_train, df_test = train_test_split(self.df, test_size=0.2, random_state=42)
		
		# Process
		print("Processing Train Data...")
		train_processed = self.fit_transform(df_train)
		print("Processing Test Data...")
		test_processed = self.transform(df_test)
		
		return train_processed, test_processed
	