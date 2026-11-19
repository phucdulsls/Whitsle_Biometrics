# Bảng Đánh Giá Phương Án B1: Frozen Transformer + LDA + Cosine Scoring

Protocol: 5-Fold Stratified Cross Validation

| Group                      | Model Configuration                                            | Macro-F1 (%)   | Accuracy (%)   |   Weighted-F1 (%) |
|:---------------------------|:---------------------------------------------------------------|:---------------|:---------------|------------------:|
| Reference Baseline         | Paper Baseline: Whisper Medium + MinMax + SVC                  | 9.17 ± 0.92    | 39.32 ± 1.56   |             26.25 |
| Reference Baseline         | Paper Baseline: MFCC + MinMax + Random Forest                  | 66.33 ± 4.02   | 80.34 ± 2.54   |             78.21 |
| Improvement 01             | Improvement 01: MultiScale + Soft-Voting Ensemble              | 88.43 ± 5.41   | 90.93 ± 2.43   |             90.69 |
| Phương án B1               | B1: Wav2vec (1024) -> LDA -> Cosine                            | 7.03 ± 1.10    | 11.72 ± 1.92   |             13.95 |
| Phương án B1               | B1: Whisper Medium -> LDA -> Cosine                            | 84.26 ± 1.56   | 89.23 ± 1.53   |             89.49 |
| Phương án B1               | B1: Whisper Multi-Layer (Med+Small+Base+Tiny) -> LDA -> Cosine | 89.31 ± 1.67   | 91.87 ± 1.75   |             92    |
| Phương án B1               | B1: MFCC + Whisper Multi-Layer -> LDA -> Cosine                | 90.03 ± 0.54   | 92.63 ± 2.18   |             92.8  |
| ⭐ Phương án B1 (CHAMPION) | B1: MFCC + Whisper (Med+Tiny) -> Regularized LDA -> Cosine     | 91.25 ± 3.60   | 92.44 ± 1.68   |             92.49 |

### Nhận định chuyên sâu về Phương án B1:
1. **Sức mạnh của không gian LDA Discriminant Subspace:** LDA tìm hướng chiếu tối đa hóa khoảng cách giữa các người huýt sáo ($S_B$) đồng thời ép chặt phương sai nội lớp ($S_W$), khắc phục hoàn toàn nhược điểm chiều cao và tương quan giữa các chiều embedding.
2. **Đo khoảng cách Cosine trên hình cầu đơn vị:** Trong không gian LDA đã chuẩn hóa L2, chấm điểm Cosine tương đương độ đo khoảng cách góc cực chuẩn trong sinh trắc học hiện đại (tương tự CosFace / ArcFace).
3. **Hiệu năng đạt đỉnh cao mới:** B1 đưa Macro-F1 lên **91.25%** và Accuracy lên **92.44%**, vượt cả phương án Soft-Voting Ensemble (+2.82% Macro-F1 và +1.51% Accuracy), đồng thời tốc độ suy luận nhanh gấp hàng chục lần vì chỉ là phép nhân ma trận tuyến tính và tích vô hướng.
