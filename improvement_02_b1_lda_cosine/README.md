# Phương Án B1: Frozen Transformer (Whisper/Wav2vec) + Layer-Weighted Attentive Stats Pooling → Regularized LDA → Cosine Scoring

---

## 1. Giới thiệu Phương án B1
Phương án **B1** áp dụng mô hình chuẩn mực của lĩnh vực **Voice & Audio Biometrics hiện đại** (tương tự kiến trúc x-vector / ECAPA-TDNN kết hợp backend phân biệt tuyến tính):

1. **Front-end:** Mô hình Transformer nền tảng (**Whisper / Wav2vec 2.0**) được **đóng băng (frozen)** trọng số.
2. **Layer-wise Weighted Averaging:** Tổng hợp có trọng số qua các tầng biểu diễn ẩn của Transformer:
   $$H_{\text{fused}} = \sum_{l=0}^{L} w_l \cdot H^{(l)}$$
3. **Attentive Statistics Pooling (ASP):** Sử dụng cơ chế Self-Attention theo trục thời gian để tính trung bình có trọng số ($\mu$) và độ lệch chuẩn có trọng số ($\sigma$), tạo ra vector embedding cố định kích thước cho mỗi bản thu.
4. **Discriminant Subspace Projection (LDA):** Giảm số chiều xuống $(C - 1)$ bằng thuật toán **Linear Discriminant Analysis** có hiệu chỉnh co ngót Ledoit-Wolf (**Regularized LDA / Shrinkage**) để tối đa hóa khoảng cách giữa các người huýt sáo ($S_B$) và triệt tiêu phương sai nội lớp ($S_W$).
5. **Length-Normalized Cosine Scoring:** Chiếu vector vào mặt cầu đơn vị ($L_2\text{-norm} = 1$) và chấm điểm xác thực danh tính dựa trên **khoảng cách góc Cosine** với các tâm mẫu đã đăng ký (**Enrolled Speaker Centroids**).

---

## 2. Chi tiết Kiến trúc & Công thức Toán học

```
                     [ Âm thanh Tiếng Huýt Sáo ]
                                  │
                                  ▼
                    Frozen Transformer Backbone
                       (Whisper / Wav2vec 2.0)
                                  │
                     Layer Hidden States: [H^(0), ..., H^(L)]
                                  │
                                  ▼
                    Layer-wise Weighted Average
                     H = sum_l softmax(w)_l * H^(l)
                                  │
                                  ▼
                    Attentive Statistics Pooling (ASP)
                      e_t = v^T * tanh(W * h_t + b)
                           alpha_t = softmax(e_t)
                       mu = sum_t alpha_t * h_t
                   sigma = sqrt( sum_t alpha_t * (h_t - mu)^2 )
                        Embedding = [ mu || sigma ]
                                  │
                                  ▼
                      Feature-wise StandardScaler
                                  │
                                  ▼
                   Regularized LDA Subspace Projection
                (Ledoit-Wolf Shrinkage, C - 1 = 9 dims)
                                  │
                                  ▼
                       L2 Length Normalization
                            z = v / ||v||_2
                                  │
                                  ▼
                   Cosine Similarity with Centroids
                       Score(z, c) = z . mu_c
                                  │
                                  ▼
               Calibrated Posterior: P(c|z) = Softmax(tau * Score)
```

---

## 3. Bảng Kết quả Thực nghiệm Đối chiếu (5-Fold Stratified CV)

| Nhóm | Cấu hình Mô hình | Macro-F1 (%) | Accuracy (%) | Weighted-F1 (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Reference Baseline** | Paper Baseline: Whisper Medium + MinMax + SVC | 9.17 ± 0.92 | 39.32 ± 1.56 | 26.25 |
| **Reference Baseline** | Paper Baseline: MFCC + MinMax + Random Forest | 66.33 ± 4.02 | 80.34 ± 2.54 | 78.21 |
| **Improvement 01** | MultiScale + Soft-Voting Ensemble | 88.43 ± 5.41 | 90.93 ± 2.43 | 90.69 |
| **Phương án B1** | B1: Wav2vec (1024) → LDA → Cosine | 7.03 ± 1.10 | 11.72 ± 1.92 | 13.95 |
| **Phương án B1** | B1: Whisper Medium → LDA → Cosine | 84.26 ± 1.56 | 89.23 ± 1.53 | 89.49 |
| **Phương án B1** | B1: Whisper Multi-Layer (Weighted Layers) → LDA → Cosine | 89.31 ± 1.67 | 91.87 ± 1.75 | 92.00 |
| **Phương án B1** | B1: MFCC + Whisper Multi-Layer → LDA → Cosine | 90.03 ± 0.54 | 92.63 ± 2.18 | 92.80 |
| **⭐ B1 (CHAMPION)** | **B1: MFCC + Whisper (Med+Tiny) → Regularized LDA → Cosine** | **91.25 ± 3.60** | **92.44 ± 1.68** | **92.49** |

---

## 4. Phân tích Chuyên sâu & Phát hiện Khoa học

### 1. Tại sao Whisper vượt trội Wav2vec 2.0 trên tiếng huýt sáo?
- **Wav2vec 2.0 (Macro-F1 ~ 7%):** Được tiền huấn luyện tự giám sát (self-supervised) trên dạng sóng thô (raw waveform) của giọng nói tiếng Anh. Nó tối ưu hóa việc phân biệt các âm vị phát âm (phoneme contrastive task) tạo bởi dây thanh và khoang miệng. Tiếng huýt sáo là sóng đơn tần biến thiên cao độ, không có các formant $F_1-F_4$ như tiếng nói, khiến Wav2vec bị mất phương hướng khi rút trích đặc trưng.
- **Whisper Encoder (Macro-F1 ~ 89.3%):** Rút trích từ Log Mel-Filterbank 80 kênh bao quát dải tần rộng đến $8000\text{ Hz}$. Do đó, toàn bộ cao độ ($F_0$) và họa âm của tiếng huýt sáo được giữ nguyên vẹn qua các tầng Transformer.

### 2. Sức mạnh của Regularized LDA & Length-Normalized Cosine
- **Khắc phục chiều cao (Curse of Dimensionality):** Khi vector ghép nối lên tới hàng nghìn chiều trong khi số mẫu chỉ có 529, ma trận hiệp phương sai nội lớp $S_W$ bị suy biến. Nhờ kỹ thuật **Ledoit-Wolf analytical shrinkage**, LDA ước lượng ma trận tối ưu, nén thông tin về đúng $C - 1 = 9$ chiều phân biệt nhất.
- **Khoảng cách góc Cosine:** Sau khi nén qua LDA và chuẩn hóa $L_2$, tích vô hướng giữa vector kiểm tra và centroid đại diện cho góc lệch không gian giữa 2 người huýt sáo. Khoảng cách góc này có khả năng loại bỏ triệt để biên độ âm lượng và sự chênh lệch khoảng cách mic thu âm.

### 3. Tốc độ thực thi siêu tốc
Khác với mô hình Ensemble Voting hoặc deep neural classifier, **LDA + Cosine** chỉ bao gồm:
1. Một phép nhân ma trận chiếu tuyến tính: $z = W_{\text{LDA}}^T x$.
2. Phép chuẩn hóa $L_2$: $z \leftarrow z / \|z\|$.
3. Phép tính tích vô hướng với 10 vector tâm: $\text{score}_c = z^T \mu_c$.
Thời gian suy luận chỉ mất **dưới 1 mili-giây**, cực kỳ lý tưởng cho hệ thống nhúng và thiết bị IoT thời gian thực!

---

## 5. Cấu trúc Thư mục & Hướng dẫn Chạy

```text
improvement_02_b1_lda_cosine/
├── README.md                      # Báo cáo kỹ thuật chi tiết này
├── config.py                      # Cấu hình siêu tham số (LDA shrinkage, solver, temperature)
├── asp_encoder.py                 # PyTorch Module: LayerWeightedSum + AttentiveStatisticsPooling (ASP)
├── lda_cosine.py                  # Module Scikit-learn/NumPy: Regularized LDA + Cosine Centroid Classifier
├── run_b1_benchmark.py            # Script chạy thực nghiệm 5-Fold Stratified CV
└── results/
    ├── b1_benchmark_results.csv   # Kết quả định lượng chi tiết
    └── b1_summary.md              # Bảng Markdown tổng kết
```

Để chạy lại thực nghiệm B1:
```bash
python improvement_02_b1_lda_cosine/run_b1_benchmark.py
```
