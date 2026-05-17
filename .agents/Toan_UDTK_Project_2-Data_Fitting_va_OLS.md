# ĐỒ ÁN 2: Data Fitting và Phương Pháp OLS

**Môn học:** Toán Ứng Dụng và Thống Kê  
**Mã môn:** MTH00051  
**Học kỳ:** HỌC KỲ 2, 2025 – 2026

**Thông tin giảng viên:**
- GV Thực hành: ThS. Võ Nam Thục Đoan, ThS. Lê Nhựt Nam
- E-mail: {vntdoan, lnnam}@fit.hcmus.edu.vn

---

## Mục lục
1. [Giới Thiệu Đồ Án](#giới-thiệu-đồ-án)
2. [Phần 1: Lý Thuyết Data Fitting và Minh Họa](#phần-1-lý-thuyết-data-fitting-và-minh-họa)
3. [Phần 2: Ứng Dụng Data Fitting vào Dữ Liệu Thực Tế](#phần-2-ứng-dụng-data-fitting-vào-dữ-liệu-thực-tế)
4. [Yêu Cầu Chung và Hướng Dẫn Nộp Bài](#yêu-cầu-chung-và-hướng-dẫn-nộp-bài)
5. [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

---

## Giới Thiệu Đồ Án

### Mục tiêu tổng quát
Đồ án này tập trung vào hai nhóm nhiệm vụ bổ sung cho nhau:
1. **Lý thuyết và minh họa** — Nắm vững nền tảng toán học của data fitting và phương pháp Ordinary Least Squares (OLS), sau đó minh họa các kết quả lý thuyết bằng code Python tự cài đặt.
2. **Ứng dụng thực tế** — Vận dụng data fitting để phân tích một bộ dữ liệu thực, bao gồm tiền xử lý, xây dựng mô hình hồi quy và đánh giá kết quả một cách có hệ thống.

### Kết quả đạt được
Sau khi hoàn thành đồ án, sinh viên có khả năng:
- Giải thích và chứng minh các tính chất cốt lõi của OLS (unbiasedness, BLUE, Gauss–Markov).
- Cài đặt pipeline data fitting hoàn chỉnh từ đầu bằng Python, có thể so sánh được với thư viện `sklearn.LinearRegression`.
- Phân tích và xử lý bộ dữ liệu thực có missing values, outliers và các vấn đề thực tiễn.
- Đánh giá mô hình một cách toàn diện (hệ số $R^2$, residual analysis, cross-validation).

### Các công cụ cho phép sử dụng
- **Python 3.10+**: Ngôn ngữ cài đặt chính.
- **NumPy, SciPy**: Tính toán số; dùng để kiểm chứng, không dùng để thay thế cài đặt thuật toán.
- **Pandas**: Đọc, xử lý và thao tác dữ liệu.
- **Matplotlib, Seaborn**: Trực quan hóa dữ liệu và kết quả mô hình.
- **Scikit-learn**: Chỉ dùng để so sánh và kiểm chứng kết quả, không dùng để cài đặt OLS chính.
- **Jupyter Notebook**: Trình bày toàn bộ thực nghiệm.

> **Lưu ý:** Các hàm như `sklearn.linear_model.LinearRegression`, `numpy.linalg.lstsq` chỉ được dùng để kiểm chứng (verification). Phần cài đặt thuật toán chính phải được viết từ đầu dựa trên công thức toán học.

---

## Phần 1: Lý Thuyết Data Fitting và Minh Họa

**Tóm tắt yêu cầu:** Trình bày lại kiến thức đã học về data fitting và OLS. Với mỗi kết quả lý thuyết, sinh viên viết code Python để minh họa và kiểm chứng bằng dữ liệu giả lập (synthetic data).

### 1.1. Bài Toán Data Fitting
#### 1.1.1. Phát biểu bài toán tổng quát
Cho tập dữ liệu $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ với $\mathbf{x}_i \in \mathbb{R}^p, y_i \in \mathbb{R}$. Bài toán data fitting là tìm hàm $f : \mathbb{R}^p \to \mathbb{R}$ trong một lớp hàm cho trước sao cho $f$ xấp xỉ tốt nhất ánh xạ từ $\mathbf{x}_i$ đến $y_i$ theo một tiêu chí đã định.

Trong mô hình hồi quy tuyến tính, ta giả thiết:
$$y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \dots + \beta_p x_{ip} + \varepsilon_i = \mathbf{x}_i^T \beta + \varepsilon_i \quad (1)$$
Viết dưới dạng ma trận:
$$\mathbf{y} = \mathbf{X}\beta + \varepsilon \quad (2)$$
Trong đó $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ là ma trận design có cột đầu toàn 1.

#### 1.1.2. Các Giả Thiết Gauss–Markov (GM1 – GM5)
- **GM1. Tuyến tính:** $\mathbf{y} = \mathbf{X}\beta + \varepsilon$.
- **GM2. Không hoàn hảo đa cộng tuyến:** $\text{rank}(\mathbf{X}) = p + 1$.
- **GM3. Ngoại sinh:** $\mathbb{E}[\varepsilon | \mathbf{X}] = 0$.
- **GM4. Đồng phương sai:** $\text{Var}(\varepsilon | \mathbf{X}) = \sigma^2 \mathbf{I}_n$.
- **GM5. Phần dư Chuẩn:** $\varepsilon | \mathbf{X} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}_n)$.

### 1.2. Phương Pháp Ordinary Least Squares (OLS)
#### 1.2.1. Hàm mất mát và nghiệm OLS
OLS tìm $\hat{\beta}$ tối thiểu hóa RSS:
$$RSS(\beta) = \|\mathbf{y} - \mathbf{X}\beta\|_2^2 = \sum_{i=1}^n (y_i - \mathbf{x}_i^T \beta)^2 \quad (3)$$
Nghiệm OLS (Normal Equations):
$$\hat{\beta}_{OLS} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y} \quad (4)$$

#### 1.2.2. Ma Trận Chiếu và Hat Matrix
Ma trận chiếu (hat matrix):
$$\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \quad (5)$$
Tính chất:
- $\mathbf{H}^2 = \mathbf{H}$ (idempotent).
- $\mathbf{H}^T = \mathbf{H}$ (đối xứng).
- Giá trị fitted: $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$; phần dư: $\hat{\varepsilon} = (\mathbf{I} - \mathbf{H})\mathbf{y}$.

#### 1.2.3. Định Lý Gauss–Markov
Dưới các giả thiết GM1–GM4, ước lượng OLS $\hat{\beta}_{OLS}$ là ước lượng tuyến tính không chệch tốt nhất (BLUE):
- Không chệch: $\mathbb{E}[\hat{\beta}_{OLS}] = \beta$.
- Tốt nhất: Có phương sai nhỏ nhất trong các ước lượng tuyến tính không chệch.
- Ma trận hiệp phương sai: $\text{Var}(\hat{\beta}_{OLS} | \mathbf{X}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$.

#### 1.2.4. Ước Lượng Phương Sai Nhiễu
$$\hat{\sigma}^2 = \frac{RSS}{n - p - 1} = \frac{\|\mathbf{y} - \mathbf{X}\hat{\beta}\|^2}{n - p - 1} \quad (7)$$

### 1.3. Đánh Giá Mô Hình
#### 1.3.1. Hệ số xác định $R^2$ và $R^2$ hiệu chỉnh
$$R^2 = 1 - \frac{RSS}{TSS} \in [0, 1] \quad (8)$$
$R^2$ hiệu chỉnh:
$$\bar{R}^2 = 1 - \frac{n - 1}{n - p - 1} (1 - R^2) \quad (9)$$

#### 1.3.2. Kiểm Định Giả Thuyết
- **t-test:** Kiểm định ý nghĩa của từng đặc trưng.
- **F-test:** Kiểm định ý nghĩa của toàn bộ mô hình.

### 1.4. Các Vấn Đề Nâng Cao
- **1.4.1. Đa cộng tuyến (Multicollinearity):** Sử dụng VIF (Variance Inflation Factor). $VIF > 10$ là nghiêm trọng.
- **1.4.2. Ridge và Lasso Regression:** Thêm thành phần chính quy hóa $L_2$ hoặc $L_1$.
- **1.4.3. Phân Tích Phần Dư (Residual Analysis):** Residuals vs Fitted, Q-Q Plot, Scale-Location, Cook's Distance.
- **1.4.4. Cross-Validation:** $k$-Fold CV. Tiêu chí lựa chọn mô hình: AIC, BIC.

### 1.5. Yêu Cầu Cài Đặt — Phần 1
Sinh viên cài đặt các hàm:
1. `ols_fit(X, y)`
2. `hat_matrix(X)`
3. `model_metrics(y, y_hat, p)`
4. `coef_inference(X, y, beta_hat, sigma2)`
5. `vif(X)`
6. `ridge_fit(X, y, lam)`
7. `residual_plots(X, y, beta_hat)`
8. `kfold_cv(X, y, k)`
9. Minh họa định lý Gauss–Markov (Monte Carlo).

---

## Phần 2: Ứng Dụng Data Fitting vào Dữ Liệu Thực Tế

**Tóm tắt yêu cầu:** Chọn ít nhất một bộ dữ liệu thực có missing values, thực hiện tiền xử lý, áp dụng data fitting để giải bài toán hồi quy, đánh giá và phân tích kết quả.

### 2.1. Tiêu Chí Chọn Bộ Dữ Liệu
- Dữ liệu thực (real-world), không dùng dữ liệu toy.
- Có ít nhất một cột bị thiếu $\ge 5\%$ dữ liệu.
- Biến mục tiêu liên tục (hồi quy).
- Kích thước: $n \ge 200$, $p \ge 3$.
- Gợi ý: Kaggle House Prices, UCI Auto MPG, World Bank Open Data, v.v.

### 2.2. Tiền Xử Lý Dữ Liệu
- **EDA:** Thống kê mô tả, phân phối, ma trận tương quan, kiểm tra trùng lắp, phân tích missing values, phát hiện outliers.
- **Xử lý Missing Values:** Listwise deletion, Mean/Median/Mode imputation, Regression imputation, k-NN imputation, MICE.
- **Các bước khác:** Feature engineering, Encoding (One-hot, Ordinal), Chuẩn hóa (z-score), Xử lý outliers.

### 2.3. Xây Dựng và Đánh Giá Mô Hình
- **Pipeline:** EDA $\to$ Tiền xử lý $\to$ Train/Test Split $\to$ Xây dựng mô hình $\to$ Đánh giá $\to$ Tinh chỉnh.
- **Các mô hình cần thử nghiệm:** OLS cơ bản, OLS chọn biến, Ridge/Lasso. (Tùy chọn: Polynomial, Kernel, Bayesian).
- **Tiêu chí so sánh:** MAE, RMSE, $R^2$ trên tập test.

### 2.4. Kỹ Thuật Nâng Cao (Tùy Chọn)
- **Kernel Regression:** Mở rộng OLS sang không gian phi tuyến qua kernel trick.
- **Bayesian Linear Regression:** Đặt prior cho $\beta$, tính phân phối hậu nghiệm.

---

## 3. Yêu Cầu Chung và Hướng Dẫn Nộp Bài

### 3.1. Cấu Trúc Báo Cáo
Báo cáo viết bằng LaTeX hoặc Markdown (xuất ra PDF), gồm: Trang bìa, Mục lục, Phần 1, Phần 2, Kết luận, Tài liệu tham khảo, Phụ lục.

### 3.2. Cấu Trúc Thư Mục Nộp Bài
```
Group_<ID>/
|-- README.md
|-- requirements.txt
|-- report/
|   |-- report.pdf
|   `-- report.tex
|-- part1/
|   |-- ols_implementation.py
|   |-- ridge_lasso.py
|   |-- residual_analysis.py
|   |-- cross_validation.py
|   `-- part1_notebook.ipynb
`-- part2/
    |-- data/
    |   `-- <ten_dataset>.csv
    |-- data_pipeline.py
    |-- model_comparison.py
    |-- advanced_methods.py
    `-- part2_notebook.ipynb
```

---

## Tài Liệu Tham Khảo
1. Gilbert Strang. *Introduction to Linear Algebra*, 6th ed. 2023.
2. Gareth James et al. *An Introduction to Statistical Learning*, 2nd ed. 2021.
3. Trevor Hastie et al. *The Elements of Statistical Learning*, 2nd ed. 2009.
4. Christopher M. Bishop. *Pattern Recognition and Machine Learning*. 2006.
5. Kevin P. Murphy. *Probabilistic Machine Learning: An Introduction*. 2022.
6. Jake VanderPlas. *Python Data Science Handbook*. 2016.
7. Wes McKinney. *Python for Data Analysis*, 3rd ed. 2022.
