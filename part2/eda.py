from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

class EDA:
	def __init__(self):
		pass

	def load_df(self, df):
		self.df = df
		self.target_col = 'RISK_MM'
		self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.to_list()
		self.numeric_cols.remove(self.target_col)
		self.categorical_cols = self.df.select_dtypes(include=['object']).columns.to_list()

	def summary_statistics(self):
		print("Summary Statistics:")
		print("="*60)
		print(self.df.describe(include='all').T)

	def normality_tests(self):
		print("--- Normality Tests ---")
		numeric_cols = self.numeric_cols
		normality_report = []

		for col in numeric_cols:
			data_clean = self.df[col].dropna()
			
			# D'Agostino-Pearson (n > 5000)
			n = len(data_clean)
			if n > 5000:
				stat, p_value = stats.normaltest(data_clean)
			else:
				stat, p_value = stats.shapiro(data_clean)
			
			is_normal = p_value > 0.05
			normality_report.append({
				'Attribute': col,
				'p-value': f"{p_value:.4f}",
				'Result': 'Normal' if is_normal else 'Non-Normal',
				'Scaler Recommendation': 'StandardScaler' if is_normal else 'MinMaxScaler/RobustScaler'
			})

		print(pd.DataFrame(normality_report))

	def correlation_analysis(self):
		print("--- Correlation Analysis ---")
		print("="*60)
		plt.figure(figsize=(16, 7))

		# Heatmap Pearson
		plt.subplot(1, 2, 1)
		pearson_corr = self.df[self.numeric_cols].corr(method='pearson')
		sns.heatmap(pearson_corr, annot=True, cmap='RdBu_r', fmt=".2f", cbar=False)
		plt.title("Pearson Correlation (Linear)")

		# Heatmap Spearman
		plt.subplot(1, 2, 2)
		spearman_corr = self.df[self.numeric_cols].corr(method='spearman')
		sns.heatmap(spearman_corr, annot=True, cmap='RdBu_r', fmt=".2f", cbar=False)
		plt.title("Spearman Correlation (Rank-based)")

		plt.tight_layout()
		plt.show()

	def duplicate_analysis(self):
		print("--- Duplicate Analysis ---")
		print("="*60)
		duplicate_count = self.df.duplicated().sum()
		print(f"Duplicate Rows: {duplicate_count}")

	def missing_values_analysis(self):
		print("--- Missing Values Analysis ---")
		print("="*60)
		missing_counts = self.df.isnull().sum()
		missing_percentages = (missing_counts / len(self.df)) * 100
		missing_df = pd.DataFrame({
			'Missing Count': missing_counts,
			'Missing Percentage': missing_percentages
		})
		print(missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False))
	
	def outlier_detection(self):
		print("--- Outlier Detection ---")
		print("="*60)

		df_numeric = self.df[self.numeric_cols]
		# get 40% of data for faster processing in outlier detection
		df_numeric = df_numeric.sample(frac=0.4, random_state=42)

		mice = IterativeImputer(random_state=42)
		df_imputed = pd.DataFrame(
			mice.fit_transform(df_numeric),
			columns=self.numeric_cols
		)

		scaler = RobustScaler()
		df_scaled = scaler.fit_transform(df_imputed)

		outlier_indices = {}

		# 1. IQR
		Q1 = df_imputed.quantile(0.25)
		Q3 = df_imputed.quantile(0.75)
		IQR = Q3 - Q1
		outlier_iqr = ((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR))).any(axis=1)
		outlier_indices['IQR'] = set(df_imputed.index[outlier_iqr])

		# 2. Z-score (ngưỡng 3)
		z_scores = np.abs(stats.zscore(df_imputed))
		outlier_z = (z_scores > 3).any(axis=1)
		outlier_indices['Z-score'] = set(df_imputed.index[outlier_z])

		# 3. Isolation Forest
		for contam in [0.01, 0.05, 0.1]:
			iso = IsolationForest(contamination=contam, random_state=42)
			preds = iso.fit_predict(df_scaled)
			outlier_indices[f'IsoForest_{contam}'] = set(df_imputed.index[preds == -1])

		# 4. Local Outlier Factor (LOF)
		for n in [10, 20, 50]:
			lof = LocalOutlierFactor(n_neighbors=n)
			preds = lof.fit_predict(df_scaled)
			outlier_indices[f'LOF_{n}'] = set(df_imputed.index[preds == -1])

		# 5. DBSCAN
		dbscan = DBSCAN(eps=2.5, min_samples=10) # eps cần tinh chỉnh tùy bộ dữ liệu
		preds = dbscan.fit_predict(df_scaled)
		outlier_indices['DBSCAN'] = set(df_imputed.index[preds == -1])

		# Outlier summary
		for method, indices in outlier_indices.items():
			print(f"{method}: {len(indices)} outliers ({len(indices)/len(df_imputed)*100:.2f}%)")

if __name__ == "__main__":
	eda = EDA()
	df = pd.read_csv("part2/weatherAUS.csv")
	eda.load_df(df)
	eda.summary_statistics()
	eda.normality_tests()
	eda.duplicate_analysis()
	eda.missing_values_analysis()
	eda.outlier_detection()