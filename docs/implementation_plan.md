# Kế hoạch Triển khai Hệ thống Sinh trắc học Âm thanh: Whistle_biometrics
*(Voice & Whistle Biometric Identification System)*

## 1. Phân tích Khoa học & Xử lý Tín hiệu Âm học (Acoustic Engineering)

### 1.1. Đặc trưng dải tần của Giọng nói (Speech) vs. Tiếng huýt sáo (Whistle)
- **Giọng người (Speech):**
  - Tần số cơ bản ($F_0$): Nam $\approx 85 - 180\text{ Hz}$, Nữ $\approx 165 - 255\text{ Hz}$, Trẻ em $\approx 250 - 400\text{ Hz}$.
  - Dải cộng hưởng (Formants $F_1, F_2, F_3, F_4$): Phổ tần kéo dài từ $300\text{ Hz}$ đến trên $4000\text{ Hz}$.
  - Đặc trưng: Dao động phức hợp đa tần từ dây thanh quản kết hợp khoang miệng, mũi.
- **Tiếng huýt sáo (Whistle / Silbo Gomero):**
  - Bản chất: Sóng âm gần như đơn tần thuần khiết (quasi-sinusoid) được tạo ra bởi luồng xoáy khí qua môi/lưỡi.
  - Dải tần tập trung: Chủ yếu từ **$500\text{ Hz}$ đến $4000\text{ Hz}$**, với cao độ biến thiên liên tục và các họa âm (harmonics) yếu hơn ở dải cao.
- **Cấu hình tối ưu chung:**
  - `SAMPLE_RATE = 16000 Hz`: Tần số Nyquist đạt $8000\text{ Hz}$, bao trùm trọn vẹn cả dải formant của giọng nói và dải tần cơ bản + họa âm của tiếng huýt sáo mà không gây lãng phí tính toán.
  - `FMIN = 50 Hz`, `FMAX = 8000 Hz`: Bao quát từ âm vực trầm của giọng người đến đỉnh cao của tiếng huýt sáo.

### 1.2. Kỹ thuật tiền xử lý (Preprocessing & VAD)
- **VAD (Voice Activity Detection) cơ bản:**
  - Tính toán năng lượng ngắn hạn (Short-Time Energy / RMS) với khung $2048$ mẫu, bước nhảy $512$ mẫu.
  - Lọc bỏ các khung có năng lượng dưới ngưỡng `VAD_ENERGY_THRESHOLD = 0.01` (hoặc ngưỡng tương đối so với peak/noise floor) để loại bỏ khoảng lặng trước và sau khi phát âm.
- **Chuẩn hóa biên độ (Amplitude Normalization):**
  - Peak Normalization: Chuẩn hóa biên độ cực đại về $0.95$ để tránh méo tiếng và triệt tiêu sự khác biệt về khoảng cách mic.

### 1.3. Biểu diễn đặc trưng (Feature Extraction Engine)
- **MFCC Backend:**
  - $N_{\text{MFCC}} = 20$ hệ số.
  - $\Delta$ (Delta - vận tốc biến thiên phổ): 20 hệ số.
  - $\Delta^2$ (Delta-Delta - gia tốc biến thiên phổ): 20 hệ số.
  - Tổng cộng 60 hệ số trên mỗi khung thời gian.
  - **Temporal Pooling:** Tính `mean` (60) và `std` (60) qua trục thời gian $\rightarrow$ **Vector 1D cố định 120 chiều** ($D = 120$) cho bất kỳ độ dài âm thanh nào.
- **Whisper Backend (Interface mở rộng):**
  - Rút trích qua Whisper Audio Encoder, trung bình pooling theo thời gian để sinh vector biểu diễn sâu.

---

## 2. Kiến trúc Hệ thống Module hóa

```text
Whitsle_biometrics/
├── config.py                  # Cấu hình tập trung (Audio, Model, Paths, VAD, Threshold)
├── src/
│   ├── audio_io.py            # Giao tiếp thu âm sounddevice, lọc khoảng lặng VAD, chuẩn hóa RMS/Peak
│   ├── features.py           # Trích xuất đặc trưng MFCC (120-dim) & Whisper backend
│   ├── enroll.py              # Đăng ký danh tính người dùng (User/Impostor/Background)
│   ├── train.py               # Huấn luyện SVM (SVC balanced + probability), xuất model.joblib
│   └── predict_live.py        # Nhận diện âm thanh trực tiếp thời gian thực
├── tests/
│   └── test_pipeline.py       # Bộ kiểm thử TDD (Unit tests với tín hiệu giả lập)
├── data/
│   ├── raw/                   # Thư mục lưu file wav đã thu âm
│   └── mock/                  # Dữ liệu synthetic giả lập cho test
├── models/
│   └── model.joblib           # Artifact mô hình đã huấn luyện
└── docs/
    └── implementation_plan.md # File kế hoạch chi tiết
```

---

## 3. Quy trình thực hiện (Phase Breakdown)

### Giai đoạn 2: Phát triển theo TDD (`tests/test_pipeline.py`)
1. `test_synthetic_signal_generator()`: Sinh sóng sine chuẩn 1000Hz và nhiễu trắng để kiểm thử không phụ thuộc phần cứng mic.
2. `test_vad_and_normalization()`: Kiểm tra VAD cắt đúng khoảng lặng và chuẩn hóa biên độ về khoảng $[-1, 1]$.
3. `test_feature_extraction_dimension()`: Kiểm tra đầu ra của `extract_features` luôn là vector 1 chiều kích thước cố định (120 chiều với MFCC).
4. `test_train_export_reload()`: Kiểm tra pipeline huấn luyện SVM với dữ liệu giả lập, lưu ra `.joblib` và load lại dự đoán chính xác.
5. `test_confidence_threshold_logic()`: Kiểm tra ngưỡng xác suất reject người lạ nếu $P(\text{User}) < \text{THRESHOLD}$.

### Giai đoạn 3: Thực thi & Lập trình chi tiết
1. `config.py`: Đặt các hằng số tiêu chuẩn và đường dẫn an toàn.
2. `src/audio_io.py`: Hàm `record_audio`, `normalize_audio`, `apply_vad`, `load_audio`, `save_audio`. Bắt ngoại lệ mic.
3. `src/features.py`: Class `AudioFeatureExtractor` hỗ trợ cả `mfcc` và `whisper`.
4. `src/enroll.py`: CLI thân thiện hướng dẫn người dùng bấm Enter để thu $N$ mẫu cho tên định danh cụ thể.
5. `src/train.py`: Đọc thư mục `data/raw/`, gom nhãn, huấn luyện `StandardScaler` + `SVC(probability=True, class_weight='balanced')`, đánh giá CV và lưu `models/model.joblib`.
6. `src/predict_live.py`: CLI cho phép hoặc thu trực tiếp từ mic hoặc truyền file wav vào để suy luận kết quả và in độ tin cậy.

### Giai đoạn 4: Kiểm thử, Debug & Xác minh
1. Chạy `pytest tests/test_pipeline.py -v`.
2. Sửa mọi lỗi sai lệch kích thước hay thư viện nếu có.
3. Tạo mock data 5 mẫu `User` và 5 mẫu `Other`, chạy `src/train.py` tạo `models/model.joblib`.
4. Chạy `src/predict_live.py --input <mock_file>` để kiểm tra độ tin cậy thực tế.
5. Nghiệm thu và hoàn tất task.

