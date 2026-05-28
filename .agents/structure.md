# PROJECT OVERVIEW: DATA FITTING & OLS (ASSIGNMENT 2)
* [cite_start]**Course:** Applied Mathematics and Statistics (Course Code: MTH00051) - Semester 2, 2025-2026[cite: 395, 396].
* [cite_start]**Institution:** University of Science - VNU-HCM[cite: 395].
* [cite_start]**Submission Deadline:** Before 23:59 on May 30, 2026, via the Faculty's Moodle system[cite: 398].

---

# I. AGENT INSTRUCTIONS

## 1. Strict Guardrails & Library Limitations
* [cite_start]**Primary Language:** Python 3.10+[cite: 399].
* [cite_start]**Mandatory Algorithm Requirements:** All regression algorithms (OLS, Ridge, Lasso), matrix operations, evaluation metrics ($MAE, RMSE, R^2$), and validation partitioning ($K\text{-Fold Cross-Validation}$) **MUST BE IMPLEMENTED FROM SCRATCH**[cite: 400, 448, 455].
* [cite_start]**Allowed Supporting Libraries:** `NumPy` and `SciPy` (Only for basic numerical computations, not to replace the core algorithm logic), `Pandas` (Data manipulation), `Matplotlib` and `Seaborn` (Data visualization)[cite: 399].
* **Scikit-Learn Constraint:** `sklearn.linear_model` and `numpy.linalg.lstsq` **ARE ONLY ALLOWED** at the very final verification steps within the Jupyter Notebooks to cross-check results. [cite_start]Do not import them into the source script (`.py`) files[cite: 400, 450, 459].

## 2. Technical & Clean Code Standards
* [cite_start]**Reproducibility:** Every function involving random data generation or data splitting must use a fixed `random_state` or `seed` parameter[cite: 418].
* [cite_start]**Unit Tests:** Every math or algorithmic function must be accompanied by at least **2 unit tests** to verify correctness on sample data[cite: 418].
* [cite_start]**Visualization:** All generated plots must include complete elements: Title, X/Y Axis Labels, and a Legend[cite: 418].
* [cite_start]**Data Leakage Prevention:** Z-score standardization parameters (`mean` and `std`) **MUST ONLY** be computed from the Train set, and then applied to `transform` both the Train and Test sets[cite: 432, 467].

---

# II. TARGET PROJECT DIRECTORY STRUCTURE
[cite_start]The Agent must construct the codebase according to this exact structural layout[cite: 416, 456]:

Group_/
├── requirements.txt
├── README.md
├── report/
│   ├── report.pdf
│   └── report.tex
├── part1/
│   ├── ols_implementation.py  # Original mathematical OLS algorithm
│   ├── ridge_lasso.py         # Ridge (Closed-form) & Lasso (Coordinate Descent) algorithms
│   ├── residual_analysis.py   # Script generating the 4 residual analysis plots
│   ├── cross_validation.py    # K-Fold Cross-Validation from scratch
│   └── part1_notebook.ipynb   # Theoretical demonstration with synthetic data
└── part2/
├── data/
│   └── <dataset_name>.csv # Real-world dataset meeting specific criteria
├── data_pipeline.py       # DataPipeline class with distinct Fit/Transform methods
├── model_comparison.py    # Evaluates MAE, RMSE, R2 and compares 3+ models
├── advanced_methods.py    # Kernel Ridge or Bayesian Regression (Optional Bonus)
└── part2_notebook.ipynb   # Complete pipeline applied to the real-world dataset


---

# III. DETAILED TASK BREAKDOWN BY PART

## PART 1: THEORY AND DEMONSTRATION (Weight: 52% - 6.0 Points) [cite: 419, 420]

### Task 1.1: Core OLS Implementation (`part1/ols_implementation.py`)
* **Mathematical Requirement:** Express the linear regression model in matrix form: $y = X\beta + \epsilon$[cite: 401, 402].
* **Function to Write:** `ols_fit(X, y)`
* **Description:** Compute estimated coefficients using the unique closed-form solution via Normal Equations: $\hat{\beta}_{OLS} = (X^TX)^{-1}X^Ty$[cite: 402, 497]. Calculate the unbiased residual variance estimator: $\hat{\sigma}^2 = \frac{RSS}{n-p-1}$.

### Task 1.2: Projection Matrix (`part1/ols_implementation.py`)
* **Function to Write:** `hat_matrix(X)`
* **Description:** Compute the Hat matrix $H = X(X^TX)^{-1}X^T$[cite: 402]. Verify its symmetric property ($H^T = H$) and idempotent property ($H^2 = H$) using assertions[cite: 402].

### Task 1.3: Metrics and Inference (`part1/ols_implementation.py`)
* **Functions to Write:** `model_metrics(y, y_hat, p)` and `coef_inference(X, y, beta_hat, sigma2)`
* **Description:**
    * Compute fundamental metrics from scratch: $RSS$, $TSS$, $R^2$, and Adjusted $R^2$[cite: 403, 455].
    * Compute the F-statistic for the overall model significance test ($F\text{-test}$)[cite: 404].
    * Calculate Standard Errors, $t\text{-statistics}$, $p\text{-values}$, and 95% confidence intervals for each coefficient $\beta_j$[cite: 404].

### Task 1.4: Multicollinearity & Regularization (`part1/ridge_lasso.py`)
* **Functions to Write:** `vif(X)`, `ridge_fit(X, y, lam)`, and `lasso_fit(X, y, lam)`
* **Description:**
    * `vif(X)`: Calculate the Variance Inflation Factor $VIF_j = \frac{1}{1-R_j^2}$ for each variable to diagnose multicollinearity[cite: 405].
    * `ridge_fit`: Solve the closed-form $L_2$ regularized solution: $\hat{\beta}_{ridge} = (X^TX + \lambda I)^{-1}X^Ty$[cite: 426, 497]. Plot the coefficient paths (Ridge Trace) against varying values of $\lambda$[cite: 426, 427].
    * `lasso_fit`: Model the loss function with an $L_1$ penalty[cite: 428]. Since Lasso lacks a closed-form solution, it must be solved iteratively from scratch using **Coordinate Descent** or **Subgradient methods**[cite: 428, 429].

### Task 1.5: Residual Analysis (`part1/residual_analysis.py`)
* **Function to Write:** `residual_plots(X, y, beta_hat)`
* **Description:** Write a script from scratch to generate exactly 4 diagnostic plots[cite: 407, 445]:
    1. *Residuals vs Fitted Plot* (Checks linearity and homoscedasticity)[cite: 445].
    2. *Normal Q-Q Plot* (Checks normality of residuals)[cite: 445].
    3. *Scale-Location Plot* (Checks constant variance)[cite: 445].
    4. *Cook's Distance Plot* (Identifies highly influential data points)[cite: 445].

### Task 1.6: Cross-Validation & Theorem Simulation (`part1/cross_validation.py`)
* **Functions to Write:** `kfold_cv(X, y, k)` and `monte_carlo_gauss_markov()`
* **Description:**
    * `kfold_cv`: Implement $k$-fold cross-validation entirely from scratch, computing the average Mean Squared Error ($MSE$) across validation folds[cite: 407, 453].
    * `monte_carlo_gauss_markov`: Design a Monte Carlo simulation program to visually demonstrate that the OLS estimator remains unbiased ($\mathbb{E}[\hat{\beta}] = \beta$) and achieves the minimum variance among linear estimators (Gauss-Markov Theorem - BLUE)[cite: 408].

---

## PART 2: REAL-WORLD APPLICATION (Weight: 48% - 5.5 Points) [cite: 419, 420]

### Dataset Selection Guardrails
The dataset loaded by the Agent into the `part2/data/` directory must strictly meet the following criteria[cite: 409, 458]:
1. Must be a real-world observational dataset (Do not use classic toy datasets such as Iris or Boston Housing from sklearn)[cite: 409].
2. Must contain at least one feature column with a missing data rate $\ge 5\%$[cite: 409, 471].
3. The target variable ($y$) must be a continuous value (Regression task)[cite: 409].
4. Minimum size constraints: Number of rows $n \ge 200$, number of feature columns $p \ge 3$[cite: 409, 471].

### Task 2.1: Exploratory Data Analysis & Processing Line (`part2/data_pipeline.py`)
* **EDA Requirement:** Print descriptive statistics, generate distributions (histograms, boxplots), display a correlation heatmap, measure missing value ratios, and filter outliers (via IQR or Z-score)[cite: 410, 464].
* **Class Architecture Design:** `DataPipeline`
* **Method Requirements:** Must implement explicit and distinct `.fit(X_train)` and `.transform(X_test)` methods[cite: 411, 432].
* **Sequential Internal Pipeline Steps:**
    1. **Missing Value Imputation:** Support configurable options such as Listwise deletion, Mean/Median/Mode imputation, Regression imputation, or k-NN imputation[cite: 464]. Justify the choice based on the missing data mechanism (MCAR, MAR, MNAR)[cite: 411].
    2. **Categorical Encoding:** Perform One-hot encoding or Ordinal encoding[cite: 411, 464].
    3. **Standardization:** Apply the Z-score calculation formula $x_j^{std} = \frac{x_j - \overline{x_j}}{s_j}$ ensuring no data leakage from the Test set occurs[cite: 430, 432].

### Task 2.2: Training Workflow & Hyperparameter Tuning (`part2/model_comparison.py`)
* **Description:** Execute K-Fold Cross-Validation (with $k=5$ or $k=10$) entirely on the scaled training dataset to determine the optimal regularization hyperparameter $\lambda$ for both Ridge and Lasso models[cite: 412, 425].

### Task 2.3: Model Evaluation & Benchmarking (`part2/model_comparison.py`)
* **Description:** Concurrently train at least 3 required models: **Baseline OLS** (all features), **Feature-Selected OLS** (excluding variables with $p\text{-value} > 0.05$ or $VIF > 10$), and the **Optimized Ridge/Lasso models**[cite: 412].
* **Output:** Output a structured evaluation table measured on the independent testing set containing three mandatory metrics: $MAE$, $RMSE$, and $R^2_{test}$[cite: 413, 434].

### Task 2.4: Feature Importance & Residual Diagnosis (`part2/model_comparison.py`)
* **Graphical Requirements:**
    1. Extract the coefficient vector $\beta$ from the best performing model and construct a horizontal or vertical bar plot (Feature Importance) to explain the physical impact of each feature variable[cite: 414, 444].
    2. Generate the set of 4 residual diagnostic plots for the best model to check for violations of the Gauss-Markov assumptions[cite: 414, 445].

### Task 2.5: Advanced Methods (Optional Bonus +0.5) (`part2/advanced_methods.py`)
* **Option 1: Kernel Ridge Regression** -> Project features into a non-linear space using a Gram matrix calculated via an RBF Kernel: $k_{RBF}(x, x') = \exp\left(-\frac{\|x-x'\|^2}{2l^2}\right)$[cite: 415].
* **Option 2: Bayesian Linear Regression** -> Establish a prior distribution $\beta \sim \mathcal{N}(