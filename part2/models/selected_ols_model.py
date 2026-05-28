from part1.ols_implementation import (
	# vif,
	ols_fit,
	coef_inference,
	ols_predict,
)
from ..utils import remove_column, vif

class SelectedOLSModel:
	def __init__(self, vif_threshold=10.0, p_value_threshold=0.05):
		self.vif_threshold = vif_threshold
		self.p_value_threshold = p_value_threshold
		self.basic_state = None
		self.selected_state = None

	def fit(self, X_train, y_train, feature_names):
		print("\n[VIF ELIMINATION]")
		print("-" * 40)
		selected_indices = list(range(len(feature_names)))
		while True:
			vif_values = vif(X_train)
			max_vif = max(vif_values)
			if max_vif <= self.vif_threshold:
				break
			max_idx = vif_values.index(max_vif)
			removed_feature = feature_names[max_idx]
			selected_indices.pop(max_idx)
			print(
				f"[REMOVE] {removed_feature:<20} "
				f"VIF = {max_vif:.4f}"
			)
			feature_names.pop(max_idx)
			X_train = remove_column(X_train, max_idx)

		print("\n[P-VALUE ELIMINATION]")
		print("-" * 40)

		while True:
			beta_hat, sigma2 = ols_fit(X_train, y_train)
			inference = coef_inference(X_train, y_train, beta_hat, sigma2)
			p_values = inference['p_value'][1:]
			max_p = max(p_values)
			if max_p <= self.p_value_threshold:
				break
			max_idx = p_values.index(max_p)
			removed_feature = feature_names[max_idx]
			selected_indices.pop(max_idx)
			print(
				f"[REMOVE] {removed_feature:<20} "
				f"p-value = {max_p:.6f}"
			)
			feature_names.pop(max_idx)
			X_train = remove_column(X_train, max_idx)

		print("\n" + "=" * 60)
		print("FINAL SELECTED FEATURES")
		print("=" * 60)

		for feature_name in feature_names:
			print(feature_name)

		print(f"\nTotal Features: {len(feature_names)}")

		beta_hat, sigma2 = ols_fit(X_train, y_train)
		self.selected_state = {
			'x': X_train,
			'y': y_train,
			'feature_names': feature_names,
			'feature_indices': selected_indices,
			'beta_hat': beta_hat,
			'sigma2': sigma2,
		}
		return beta_hat, sigma2

	def predict(self, X_test):
		if self.selected_state is None:
			raise ValueError("Call fit_selected() before predict().")

		X_selected_test = [] 
		for row in X_test: 
			new_row = [ row[idx] for idx in self.selected_state['feature_indices'] ]
			X_selected_test.append(new_row)

		y_pred = ols_predict(X_selected_test, self.selected_state['beta_hat'])
		
		return y_pred