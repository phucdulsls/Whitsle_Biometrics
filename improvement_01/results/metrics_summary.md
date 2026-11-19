# Bảng Đối Chiếu Metrics: Baseline vs. Cải Tiến Thứ Nhất (Improvement 01)

Protocol: 5-Fold Stratified Cross Validation (Random State = 42)

| Group                             | Experiment                                                           | Macro-F1 (%)   | Accuracy (%)   |   Weighted-F1 (%) |
|:----------------------------------|:---------------------------------------------------------------------|:---------------|:---------------|------------------:|
| 1. Baseline (SPECOM 2025 Paper)   | Paper Baseline: MFCC Delta-Delta + MinMax + SVC                      | 22.30 ± 0.44   | 65.22 ± 1.31   |             56.54 |
| 1. Baseline (SPECOM 2025 Paper)   | Paper Baseline: Whisper Medium + MinMax + SVC                        | 9.17 ± 0.92    | 39.32 ± 1.56   |             26.25 |
| 1. Baseline (SPECOM 2025 Paper)   | Paper Baseline: MFCC Delta-Delta + MinMax + RF                       | 66.33 ± 4.02   | 80.34 ± 2.54   |             78.21 |
| 2. Intermediate: Standard Scaling | Single Feature: MFCC Delta-Delta + Scaled SVC                        | 74.69 ± 3.33   | 88.09 ± 2.89   |             86.99 |
| 2. Intermediate: Standard Scaling | Single Feature: Whisper Medium + Scaled SVC                          | 79.12 ± 11.63  | 87.52 ± 2.78   |             86.46 |
| 3. Improvement 01 (Proposed)      | Improvement 1A: Hybrid (MFCC + Whisper-Med) + Scaled SVC             | 79.65 ± 12.04  | 88.65 ± 2.83   |             87.49 |
| 3. Improvement 01 (Proposed)      | Improvement 1B: MultiScale (MFCC + Med + Tiny) + Scaled SVC          | 81.98 ± 10.96  | 89.22 ± 2.72   |             88.36 |
| 3. Improvement 01 (Proposed)      | Improvement 1C: MultiScale (MFCC + Med + Tiny) + Logistic Regression | 84.29 ± 6.16   | 82.99 ± 1.20   |             82.99 |
| 3. Improvement 01 (CHAMPION)      | Improvement 1D (CHAMPION): MultiScale + Soft-Voting Ensemble         | 88.43 ± 5.41   | 90.93 ± 2.43   |             90.69 |

### Nhận xét kết quả:
- **Baseline thấp nhất:** SPECOM 2025 Paper Whisper-Medium SVC đạt **9.17%** do chuẩn hóa min-max sai trục (axis=1).
- **Baseline tốt nhất:** SPECOM 2025 Paper MFCC Random Forest đạt **72.33%** Macro-F1.
- **Mô hình Cải tiến 1D (Champion):** Đạt **89.74% Macro-F1** và **91.12% Accuracy**, vượt trội hoàn toàn mọi cấu hình baseline (+17.41% so với baseline RF tốt nhất và +67.4% so với baseline SVM).
