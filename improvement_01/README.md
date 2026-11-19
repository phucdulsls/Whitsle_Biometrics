# Cải Tiến Thứ Nhất: Multi-Scale Acoustic-Deep Feature Fusion & Ensemble Soft Voting
*(Improvement 01: Multi-Scale Feature Fusion & Robust Calibrated Ensemble for Whistler Biometrics)*

---

## 1. Mục tiêu Cải tiến (Objective)
Giải quyết triệt để các điểm nghẽn hiệu năng của hệ thống Baseline (bài báo SPECOM 2025 và pipeline đơn đặc trưng), thiết lập mốc chuẩn mới (**New State-of-the-Art Benchmark**) trên tập dữ liệu tiếng huýt sáo (Silbo Gomero, 10 đối tượng, 529 mẫu với sự mất cân bằng dữ liệu cực lớn từ 7 đến 183 mẫu/lớp).

**Mục tiêu định lượng:**
- **Vượt xa Macro-F1 của Baseline:** Nâng Macro-F1 từ **22.30%** (Paper Baseline SVM) và **66.33%** (Paper Baseline RF) lên **trên 88%**.
- **Đạt độ chính xác toàn diện (Overall Accuracy):** Vượt mốc **90%** (tăng từ 65.22% / 80.34%).
- **Cân bằng hiệu năng trên các lớp thiểu số:** Các lớp chỉ có 7-8 mẫu (Class 1, 4) không còn bị mô hình bỏ quên hoặc phân loại nhầm về lớp đa số (Class 6: 183 mẫu).

---

## 2. Phân tích Nguyên nhân Điểm nghẽn ở Baseline (Root Cause Analysis)

1. **Sai lầm trong kỹ thuật chuẩn hóa mẫu (Sample-wise Min-Max):**
   - Trong `src/train_classifier.py` của bài báo, hàm `min_max_normalization(X)` áp dụng `axis=1` (chuẩn hóa trên từng dòng/mẫu đơn lẻ):
     $$\tilde{x}_i = \frac{x_i - \min(x_i)}{\max(x_i) - \min(x_i)}$$
   - Điều này làm triệt tiêu sự chênh lệch biên độ năng lượng đặc trưng giữa các dải tần số âm thanh, khiến mô hình SVM trên Whisper bị giảm sút nghiêm trọng (chỉ đạt **9.17%** Macro-F1).
2. **Thiếu sự kết hợp đa tầng đặc trưng (Single-modal limitation):**
   - Tiếng huýt sáo là sóng âm gần như đơn tần thuần khiết (quasi-sinusoid) biến thiên tần số rất nhanh theo thời gian:
     - **MFCC ($\Delta + \Delta^2$):** Bắt rất nhạy các biến thiên tần số ngắn hạn, vận tốc phổ và gia tốc phổ của tiếng huýt sáo.
     - **Whisper Encoders:** Được huấn luyện trước trên kho ngữ liệu khổng lồ, nắm bắt cấu trúc chuỗi thời gian tổng thể và đặc trưng âm học ngữ nghĩa sâu.
   - Baseline chỉ chạy từng đặc trưng đơn lẻ mà không có cơ chế **Feature Fusion**.
3. **Mất cân bằng dữ liệu nghiêm trọng (Severe Imbalance 1:26):**
   - Tỷ lệ mẫu giữa lớp lớn nhất (183 mẫu) và lớp nhỏ nhất (7 mẫu) lên tới hơn 26 lần. Việc áp dụng SMOTE thông thường với số láng giềng cưỡng bức $k \le 4$ trong không gian số chiều cao (1024 chiều) dễ sinh nhiễu ngoại lai.

---

## 3. Kiến trúc Cải tiến (Proposed Architecture)

```
                            [ RAW AUDIO / FEATURES ]
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
  MFCC Delta-Delta               Whisper Medium                  Whisper Tiny
   (60 dimensions)              (1024 dimensions)              (384 dimensions)
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                       Multi-Scale Feature Fusion
                         Vector: 1468 dimensions
                                       │
                                       ▼
                         Feature-wise StandardScaler
                                       │
                                       ▼
                            L2 Unit Hypersphere Norm
                         (Cosine Geometric Embedding)
                                       │
                                       ▼
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
 Calibrated RBF-SVC             Regularized Logistic           Multi-Layer Perceptron
 (Balanced Class Penalty)        Regression (Balanced)                 (MLP)
   (Weight: 2.0)                   (Weight: 1.5)                   (Weight: 1.0)
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                        Soft-Voting Ensemble Decision
                                       │
                                       ▼
                [ OUTPUT: Predicted Whistler ID & Probability ]
```

### Các thành phần chính:
1. **Multi-Scale Feature Fusion (1468 chiều):**
   - Kết hợp bổ trợ: $60\text{ dims (MFCC } \Delta+\Delta^2) + 1024\text{ dims (Whisper Medium)} + 384\text{ dims (Whisper Tiny)}$.
   - Đảm bảo nắm bắt trọn vẹn cả chi tiết tần số cơ bản của tiếng huýt sáo lẫn biểu diễn nơ-ron sâu đa tỉ lệ.
2. **Feature-wise Standard Scaling & L2 Hypersphere Normalization:**
   - Chuẩn hóa z-score theo từng đặc trưng qua toàn bộ tập dữ liệu, sau đó chiếu các vector về mặt cầu đơn vị ($L_2\text{-norm} = 1$).
   - Trong không gian chuẩn hóa $L_2$, tích vô hướng tương đương với độ tương đồng Cosine, tối ưu hóa tuyệt vời cho các bài toán phân biệt sinh trắc học.
3. **Cost-Sensitive Multi-Model Ensemble Soft Voting:**
   - Thay vì phụ thuộc vào một bộ phân loại duy nhất, ta kết hợp 3 mô hình học máy bổ trợ nhau với xác suất hiệu chuẩn:
     - **SVC (RBF Kernel, $C=10.0$, `class_weight='balanced'`):** Tìm siêu phẳng phân tách phi tuyến với biên độ cách ly lớn nhất và phạt nặng lỗi trên lớp thiểu số.
     - **Multinomial Logistic Regression ($C=1.0$, `class_weight='balanced'`):** Tuyến tính, chống overfit cực tốt trên không gian số chiều cao.
     - **Multi-Layer Perceptron (MLP 128 hidden units):** Học các tương tác phi tuyến tính phức tạp giữa các đặc trưng ghép nối.
   - Cơ chế Soft-Voting tổng hợp trọng số xác suất dự đoán giúp hạ thấp phương sai (variance) và nâng cao độ tin cậy.

---

## 4. Kết quả Thực nghiệm & Bảng Đối Chiếu Metrics

Đánh giá thực nghiệm khách quan theo giao thức chuẩn **5-Fold Stratified Cross Validation** (Random Seed = 42) trên toàn bộ 529 mẫu:

| Nhóm Thử Nghiệm | Cấu hình Mô hình | Macro-F1 (%) | Accuracy (%) | Weighted-F1 (%) |
| :--- | :--- | :---: | :---: | :---: |
| **1. Baseline (SPECOM 2025 Paper)** | MFCC Delta-Delta + MinMax + SVC | 22.30 ± 0.44 | 65.22 ± 1.31 | 56.54 |
| **1. Baseline (SPECOM 2025 Paper)** | Whisper Medium + MinMax + SVC | 9.17 ± 0.92 | 39.32 ± 1.56 | 26.25 |
| **1. Baseline (SPECOM 2025 Paper)** | MFCC Delta-Delta + MinMax + Random Forest | 66.33 ± 4.02 | 80.34 ± 2.54 | 78.21 |
| **2. Trung gian (Đơn đặc trưng + Scaled)** | MFCC Delta-Delta + Scaled SVC | 74.69 ± 3.33 | 88.09 ± 2.89 | 86.99 |
| **2. Trung gian (Đơn đặc trưng + Scaled)** | Whisper Medium + Scaled SVC | 79.12 ± 11.63 | 87.52 ± 2.78 | 86.46 |
| **3. Cải Tiến 1A (Hybrid Fusion)** | Fused (MFCC + Whisper-Med) + Scaled SVC | 79.65 ± 12.04 | 88.65 ± 2.83 | 87.49 |
| **3. Cải Tiến 1B (MultiScale Fusion)** | Fused (MFCC + Med + Tiny) + Scaled SVC | 81.98 ± 10.96 | 89.22 ± 2.72 | 88.36 |
| **3. Cải Tiến 1C (Linear Soft Margin)** | Fused (MFCC + Med + Tiny) + Logistic Reg | 84.29 ± 6.16 | 82.99 ± 1.20 | 82.99 |
| **⭐ 3. Cải Tiến 1D (CHAMPION ENSEMBLE)** | **MultiScale Fusion + Soft-Voting Ensemble** | **88.43 ± 5.41** | **90.93 ± 2.43** | **90.69** |

### Đánh giá định lượng:
- **Macro-F1:** Đạt **88.43%**, tăng **+66.13%** so với baseline SVM của bài báo và tăng **+22.10%** so với baseline Random Forest tốt nhất của bài báo.
- **Accuracy:** Đạt **90.93%**, chính thức đưa hệ thống nhận dạng tiếng huýt sáo vượt ngưỡng 90%.
- **Độ ổn định:** Weighted-F1 đạt **90.69%**, chứng minh mô hình không chỉ đoán tốt lớp đa số mà phân loại xuất sắc trên toàn bộ 10 định danh.

---

## 5. Cấu trúc Thư mục `improvement_01/`

```text
improvement_01/
├── README.md                      # Báo cáo kỹ thuật chi tiết này
├── config.py                      # Cấu hình đường dẫn, tham số thực nghiệm
├── data_loader.py                 # Module nạp & hợp nhất đa tầng đặc trưng (Multi-Scale Fusion)
├── models.py                      # Định nghĩa Pipeline chuẩn hóa & Ensemble Soft Voting
├── evaluate.py                    # Khung đánh giá Stratified 5-Fold Cross Validation
├── run_benchmark.py               # Script tự động thực thi và xuất bảng so sánh
└── results/
    ├── benchmark_comparison.csv   # Kết quả thô dạng CSV
    └── metrics_summary.md         # Bảng tổng kết số liệu Markdown
```

---

## 6. Hướng dẫn Chạy Thực nghiệm (Reproduction)

Để chạy lại toàn bộ benchmark đối chứng và cập nhật bảng metrics:

```bash
python improvement_01/run_benchmark.py
```
Kết quả chi tiết sẽ được tự động xuất ra file [`improvement_01/results/benchmark_comparison.csv`](file:///d:/Whitsle_biometrics/improvement_01/results/benchmark_comparison.csv) và [`improvement_01/results/metrics_summary.md`](file:///d:/Whitsle_biometrics/improvement_01/results/metrics_summary.md).
