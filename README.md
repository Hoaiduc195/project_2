# Dự án Thống kê Ứng dụng: Hồi quy Tuyến tính, Data Fitting & Pipeline Đánh giá Mô hình

Dự án này bao gồm việc triển khai, kiểm chứng và đánh giá hiệu năng (benchmarking) của các thuật toán hồi quy tuyến tính và điều chuẩn từ đầu (sử dụng Python thuần và NumPy) như một phần của nghiên cứu Thống kê Ứng dụng. Dự án được chia làm hai phần:

* **Phần 1**: Dẫn giải lý thuyết và tự triển khai OLS, Ridge, Lasso, chẩn đoán mô hình và mô phỏng Monte Carlo kiểm chứng định lý Gauss-Markov.
* **Phần 2**: Pipeline tiền xử lý lưu trạng thái (stateful data pipeline), lựa chọn đặc trưng lặp (iterative feature selection) và đánh giá mô hình trên tập dữ liệu thời tiết Úc (`weatherAUS.csv`).

---

## 1. Tóm tắt Dự án

### Phần 1: Lý thuyết Hồi quy Tuyến tính & Triển khai Tự viết
* **Các Phép toán Ma trận bằng Python thuần**: Triển khai các phép chuyển vị ma trận, nhân ma trận, nghịch đảo ma trận và các hàm bổ trợ từ đầu mà không dùng thư viện bên thứ ba.
* **Phương pháp Bình phương Tối thiểu Cổ điển (OLS)**: Giải hệ phương trình chuẩn tắc, tính toán sai số chuẩn (standard error), thống kê $t$, thống kê $F$ và p-value hoàn toàn tự cài đặt.
* **Tính chất Ma trận Hat**: Kiểm chứng bằng toán học và thực nghiệm số các tính chất đối xứng ($H^T = H$), lũy đẳng ($H^2 = H$) và vết ($Tr(H) = p + 1$).
* **Điều chuẩn tự viết**: Triển khai Hồi quy Ridge (nghiệm dạng đóng với phạt L2) và Hồi quy Lasso (Coordinate Descent với phạt L1) từ đầu, bao gồm cả vẽ biểu đồ vết hệ số (coefficient trace plot).
* **Mô phỏng Gauss-Markov Monte Carlo**: Mô phỏng kiểm chứng định lý Gauss-Markov chứng minh OLS là ước lượng tuyến tính không chệch tốt nhất (BLUE) dưới điều kiện phương sai sai số thay đổi.
* **Chẩn đoán Mô hình**: Tự vẽ bộ 4 biểu đồ chẩn đoán phần dư chuẩn (Residuals vs Fitted, Normal Q-Q, Scale-Location, Cook's Distance) từ đầu.

### Phần 2: Pipeline Tiền xử lý Stateful & Đánh giá Mô hình
* **Pipeline Tiền xử lý Stateful**: Xây dựng pipeline tiền xử lý lưu trữ trạng thái để xử lý điền khuyết, mã hóa one-hot và chuẩn hóa Z-score trên tập dữ liệu thời tiết để tránh rò rỉ dữ liệu (data leakage).
* **Lựa chọn Đặc trưng**: Triển khai lọc bớt đặc trưng tự động dựa trên Hệ số phóng đại phương sai (VIF > 10) và mức ý nghĩa thống kê của hệ số (p-value > 0.05).
* **Đánh giá Mô hình**: So sánh OLS cơ bản, OLS đã lọc đặc trưng, Ridge tối ưu (tìm tham số bằng kiểm chứng chéo 5-fold) và Lasso tối ưu để dự báo biến mục tiêu `MaxTemp`.

---

## 2. Cấu trúc Dự án & Cách dùng File

Dưới đây là sơ đồ cây thư mục ánh xạ các tệp tin và chức năng của chúng:

```
project_2/
│
├── README.md                      # Tài liệu hướng dẫn dự án (tệp tin này)
├── requirements.txt               # Các thư viện và phụ thuộc cần thiết
│
├── part1/                         # Lý thuyết cốt lõi & các triển khai tự viết
│   ├── matrix_ops.py              # Các hàm đại số ma trận Python tự cài đặt
│   ├── ols_implementation.py      # Ước lượng OLS, chỉ số mô hình, VIF và Monte Carlo
│   ├── ridge_lasso.py             # Ước lượng Ridge và Lasso, hàm vẽ đồ thị vết hệ số
│   ├── residual_analysis.py       # Hàm vẽ biểu đồ chẩn đoán phần dư
│   ├── cross_validation.py        # Thuật toán kiểm chứng chéo K-Fold tự cài đặt
│   ├── part1_notebook.ipynb       # Notebook thực nghiệm cho Phần 1
│   └── test_part1.py              # Bộ kiểm thử đơn vị (unittest) cho Phần 1
│
└── part2/                         # Pipeline Stateful & Đánh giá hiệu năng
    ├── data/
    │   └── weatherAUS.csv         # Tập dữ liệu thời tiết gốc
    ├── data_pipeline.py           # Lớp pipeline tiền xử lý DataPipeline
    ├── model_comparison.py        # Động cơ so sánh mô hình, lựa chọn đặc trưng, tối ưu tham số
    ├── part2_notebook.ipynb       # Notebook thực nghiệm cho Phần 2
    ├── model_comparison_results.csv # Bảng chỉ số đánh giá mô hình được xuất ra
    └── best_model_diagnostics.png # Biểu đồ chẩn đoán được vẽ cho mô hình tốt nhất
```

### Chi tiết các thành phần
* **[matrix_ops.py](file:///d:/Applied%20Statistics/project_2/part1/matrix_ops.py)**: Triển khai các phép toán đại số tuyến tính cơ bản (nhân ma trận, nghịch đảo ma trận bằng phương pháp khử Gauss-Jordan, chuyển vị và thêm cột intercept).
* **[ols_implementation.py](file:///d:/Applied%20Statistics/project_2/part1/ols_implementation.py)**: Chứa hàm tự viết `ols_fit`, kiểm định giả thuyết hệ số và bộ mô phỏng Monte Carlo kiểm chứng định lý Gauss-Markov.
* **[ridge_lasso.py](file:///d:/Applied%20Statistics/project_2/part1/ridge_lasso.py)**: Triển khai thuật toán Ridge (qua Normal Equations) và Lasso (qua Coordinate Descent) hỗ trợ truyền một hoặc nhiều tham số lambda, tự động định dạng và vẽ biểu đồ vết hệ số.
* **[data_pipeline.py](file:///d:/Applied%20Statistics/project_2/part2/data_pipeline.py)**: Đảm nhận nhiệm vụ tải dữ liệu, làm sạch biến mục tiêu, chia nhỏ dữ liệu và chạy luồng huấn luyện.
* **[model_comparison.py](file:///d:/Applied%20Statistics/project_2/part2/model_comparison.py)**: Tích hợp pipeline tiền xử lý, thực hiện lựa chọn đặc trưng lặp qua VIF & p-value, tối ưu hóa Ridge/Lasso bằng kiểm chứng chéo K-fold và lưu lại các chỉ số đánh giá kèm biểu đồ chẩn đoán.

---

## 3. Thiết lập Môi trường

Các thư viện phụ thuộc được liệt kê trong [requirements.txt](file:///d:/Applied%20Statistics/project_2/requirements.txt). Một môi trường ảo có tên `venv` nằm ở thư mục gốc đã được cài đặt sẵn tất cả các gói thư viện.

Cách kích hoạt môi trường ảo đã cấu hình sẵn trên Windows:
```powershell
.\venv\Scripts\Activate.ps1
```

Cách cài đặt thủ công các gói thư viện vào môi trường ảo khác:
```bash
pip install -r requirements.txt
```

---

## 4. Hướng dẫn Chạy Dự án

### Chạy trên Jupyter Notebooks (`.ipynb`)
Mở môi trường Jupyter hoặc IDE của bạn (chẳng hạn như VS Code), chọn kernel tương ứng với môi trường ảo **`venv`** và chạy các ô lệnh theo thứ tự tuần tự:
1. **Notebook Phần 1**: Mở [part1_notebook.ipynb](file:///d:/Applied%20Statistics/project_2/part1/part1_notebook.ipynb) để xem kiểm chứng toán học, suy diễn OLS, mô phỏng Monte Carlo và biểu đồ vết Ridge/Lasso.
2. **Notebook Phần 2**: Mở [part2_notebook.ipynb](file:///d:/Applied%20Statistics/project_2/part2/part2_notebook.ipynb) để xem tiền xử lý dữ liệu, lựa chọn đặc trưng lặp và so sánh các mô hình.

### Chạy bằng Python thuần (`.py`)

#### 1. Chạy các bài kiểm thử đơn vị (Unit Tests)
Bạn có thể kiểm chứng các triển khai toán học và hàm tự viết từ đầu bằng cách chạy bộ kiểm thử bên trong thư mục `part1/`:
```bash
# Di chuyển đến thư mục part1
cd part1
# Chạy unit tests
python -m unittest test_part1.py
```

#### 2. Chạy kịch bản đánh giá mô hình (Benchmarking)
Để chạy so sánh đánh giá hiệu năng từ đầu đến cuối trên tập dữ liệu thời tiết:
```bash
# Di chuyển đến thư mục part2
cd part2
# Chạy kịch bản so sánh đánh giá
python model_comparison.py
```
Kịch bản này sẽ in ra các bước lựa chọn đặc trưng, tối ưu hóa tham số mô hình điều chuẩn, in bảng chỉ số đánh giá cuối cùng và tạo/lưu các biểu đồ chẩn đoán phần dư tại [best_model_diagnostics.png](file:///d:/Applied%20Statistics/project_2/part2/best_model_diagnostics.png).
