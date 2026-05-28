import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    # Handle division by zero / infinite VIF issues
    try:
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    except:
        vif_data["VIF"] = np.inf
    return vif_data