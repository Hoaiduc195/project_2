import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from scipy.stats import chi2
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, roc_auc_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer, StandardScaler
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import category_encoders as ce
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import umap

class DataPipeline:
	def __init__(self, data_source):
		self.data_source = data_source
		self.num_imputer = IterativeImputer(random_state=42, max_iter=50)
		self.cat_imputer = SimpleImputer(strategy='most_frequent')
		self.cat_cols_to_impute = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
		self.bin_encoder = ce.BinaryEncoder(cols=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
		self.target_encoder = ce.TargetEncoder(
			cols=['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'],
			smoothing=10 
		)

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
			df = self.target_encoder.fit_transform(df, df['RISK_MM'])
		else:
			# df_target_encoded = self.target_encoder.transform(df)
			# df_bin_encoded = self.bin_encoder.transform(df[self.bin_cols])
			# df = pd.concat([df, df_target_encoded, df_bin_encoded], axis=1)
			df = self.target_encoder.transform(df)

		# Drop original categorical columns after encoding
		cols_to_drop = ['Date', 'Month', 'RainToday', 'RainTomorrow']
		df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

		return df

	def extract(self):
		# Code to extract data from the source
		self.df = pd.read_csv(self.data_source)
		self.df = self.df.dropna(subset=['RISK_MM']).reset_index(drop=True)
		self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.to_list()
		self.numeric_cols.remove('RISK_MM')
		self.categorical_cols = self.df.select_dtypes(include=['str']).columns.to_list()
		self.numeric_df = self.df[self.numeric_cols]
		self.categorical_df = self.df[self.categorical_cols]

	def fit_transform(self, df):
		df_num = df[self.numeric_cols].copy()
		df_cat = df[self.categorical_cols].copy()
		target = df['RISK_MM'].copy()

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
		target = df['RISK_MM'].copy()

		# 1. Imputation using the fitted imputers
		df_num = self._numeric_missing_imputation(df_num, is_train=False)
		df_cat = self._categorical_missing_imputation(df_cat, is_train=False)

		df_combined = pd.concat([df_num, df_cat, target], axis=1)
		
		# Encoding using the fitted encoders
		df_combined = self._categorical_encoding(df_combined, is_train=False)
		return df_combined

	def data_info(self, df):
		print("Data Shape:", df.shape)
		print("\nMissing Values:\n", df.isnull().sum())
		print("\nData Types:\n", df.dtypes)
		print("\nStatistical Summary:\n", df.describe())
		print("\nNumeric Columns:\n", df.select_dtypes(include=[np.number]).columns)
		print("\nCategorical Columns:\n", df.select_dtypes(include=['str']).columns)

	def get_processed_data(self):
		self.extract()
		
		# Split Train / Test
		df_train, df_test = train_test_split(self.df, test_size=0.2, random_state=42)
		
		# Process
		print("Train...")
		train_processed = self.fit_transform(df_train)
		print("Test...")
		test_processed = self.transform(df_test)
		
		return train_processed, test_processed

	def run(self):
		'''
		TODO:
		'''
		train_data, test_data = self.get_processed_data()
		print("Train Data Processed Shape:", train_data.shape)
		print("Test Data Processed Shape:", test_data.shape)
		print("\nTrain Data Info:")
		self.data_info(train_data)
		print("\nTest Data Info:")
		self.data_info(test_data)

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    # Handle division by zero / infinite VIF issues
    try:
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    except:
        vif_data["VIF"] = np.inf
    return vif_data

def run_ols_models(train_data, test_data):
    print("\n" + "="*60)
    print("Model 1: Base OLS (ALL FEATURES)")
    print("="*60)
    
    y_train = train_data['RISK_MM']
    X_train = train_data.drop(columns=['RISK_MM'])
    X_train_sm = sm.add_constant(X_train)
    ols_basic = sm.OLS(y_train, X_train_sm).fit()
    
    print("\n" + "="*60)
    print("Model 2: OLS Variable Selection (Dropping VIF > 10 and P-Value > 0.05)")
    print("="*60)
    
    X_selected = X_train.copy()
    
    # VIF check
    while True:
        vif_df = calculate_vif(X_selected)
        max_vif = vif_df['VIF'].max()
        if max_vif > 10:
            feature_to_drop = vif_df.loc[vif_df['VIF'] == max_vif, 'Feature'].iloc[0]
            X_selected = X_selected.drop(columns=[feature_to_drop])
        else:
            break
            
    # p-value check
    while True:
        X_selected_sm = sm.add_constant(X_selected)
        ols_step = sm.OLS(y_train, X_selected_sm).fit()
        p_values = ols_step.pvalues.drop('const', errors='ignore') 
        max_p_value = p_values.max()
        
        if max_p_value > 0.05:
            feature_to_drop = p_values.idxmax()
            X_selected = X_selected.drop(columns=[feature_to_drop])
        else:
            break
            
    X_final_sm = sm.add_constant(X_selected)
    ols_final = sm.OLS(y_train, X_final_sm).fit()
    print(f"Final Model Features: {len(X_selected.columns)} variables.")

	# Test evaluation
    print("\n" + "="*60)
    print("Test Evaluation")
    print("="*60)
    
    y_test = test_data['RISK_MM']
    
    X_test = test_data[X_selected.columns]
    
    X_test_sm = sm.add_constant(X_test, has_constant='add')
    
    y_pred = ols_final.predict(X_test_sm)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"1. MAE (Mean Absolute Error) : {mae:.4f} mm")
    print(f"2. RMSE (Root Mean Squared Error): {rmse:.4f} mm")
    print(f"3. R-squared (R²)              : {r2:.4f}")

if __name__ == "__main__":
    data_pipeline = DataPipeline(data_source='part2/weatherAUS.csv') 
    train_data, test_data = data_pipeline.get_processed_data()
    
    run_ols_models(train_data, test_data)