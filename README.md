# Whistle Biometrics

Hệ thống nhận dạng sinh trắc học âm thanh cho tiếng huýt sáo (Whistler Identification / Silbo Gomero) và nhận diện trực tiếp qua micro, gồm pipeline thu âm thời gian thực và hai nhánh cải tiến trên tập đặc trưng âm học/deep embeddings.

---

## Cấu trúc dự án

```text
Whitsle_Biometrics/
├── config.py                         # Cấu hình âm học (16 kHz, VAD, MFCC, SVM)
├── requirements.txt                  # Thư viện Python cần thiết
├── src/                              # Pipeline sinh trắc học thời gian thực
│   ├── audio_io.py                   # Thu âm, chuẩn hóa biên độ, VAD cắt khoảng lặng
│   ├── features.py                   # Trích xuất MFCC (120-dim) & Whisper embedding
│   ├── enroll.py                     # Đăng ký mẫu người dùng
│   ├── train.py                      # Huấn luyện mô hình SVM (StandardScaler + Balanced SVC)
│   └── predict_live.py               # Nhận diện trực tiếp từ micro hoặc file .wav
├── improvement_01/                   # Cải tiến 1: Multi-Scale Feature Fusion + Soft-Voting Ensemble
│   ├── README.md                     # Báo cáo kỹ thuật & bảng kết quả chi tiết
│   ├── run_benchmark.py              # Script đánh giá 5-Fold Stratified CV
│   └── results/                      # Kết quả CSV & Markdown
├── improvement_02_b1_lda_cosine/     # Cải tiến 2 (B1): Frozen Transformer + Regularized LDA + Cosine
│   ├── README.md                     # Báo cáo kỹ thuật & bảng kết quả chi tiết
│   ├── asp_encoder.py                # Layer-weighted sum + Attentive Statistics Pooling (PyTorch)
│   ├── lda_cosine.py                 # Regularized LDA (Ledoit-Wolf) + Cosine Centroid Scoring
│   ├── run_b1_benchmark.py           # Script đánh giá 5-Fold Stratified CV
│   └── results/                      # Kết quả CSV & Markdown
├── docs/implementation_plan.md       # Kế hoạch triển khai hệ thống
└── tests/test_pipeline.py            # Bộ unit test (pytest)
```

---

## Tóm tắt kết quả thực nghiệm (5-Fold Stratified CV)

| Phương pháp | Macro-F1 (%) | Accuracy (%) | Weighted-F1 (%) |
| :--- | :---: | :---: | :---: |
| **Baseline đối chứng** (MFCC $\Delta\Delta$ + Sample-wise MinMax + RF) | 66.33 ± 4.02 | 80.34 ± 2.54 | 78.21 |
| **Cải tiến 1** (Multi-Scale Fusion + Soft-Voting Ensemble) | 88.43 ± 5.41 | 90.93 ± 2.43 | 90.69 |
| **Cải tiến 2 – B1** (MFCC + Whisper Med/Tiny + Regularized LDA + Cosine) | **91.25 ± 3.60** | **92.44 ± 1.68** | **92.49** |

Chi tiết kiến trúc và các cấu hình thử nghiệm xem tại:
- [improvement_01/README.md](improvement_01/README.md)
- [improvement_02_b1_lda_cosine/README.md](improvement_02_b1_lda_cosine/README.md)

---

## Cài đặt & Chạy thử

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Chạy kiểm thử đơn vị (Unit Tests)
```bash
pytest tests/test_pipeline.py -v
```

### 3. Chạy benchmark Cải tiến 1 & Cải tiến 2
> **Lưu ý về dữ liệu đặc trưng (`features/`):** Repository này **không phân phối lại** dữ liệu đặc trưng trích xuất sẵn từ bài báo gốc. Để tái lập các thí nghiệm benchmark, vui lòng tải thư mục `features/` trực tiếp từ repository chính thức của tác giả bài báo SPECOM 2025:
> https://github.com/jose-jvmas/WhistlerIdentificationSilbo
>
> Sau đó đặt thư mục `features/` vào thư mục gốc của dự án và chạy:

```bash
python improvement_01/run_benchmark.py
python improvement_02_b1_lda_cosine/run_b1_benchmark.py
```

### 4. Sử dụng pipeline sinh trắc học trực tiếp
```bash
python src/enroll.py
python src/train.py
python src/predict_live.py
```

---

## Tài liệu tham khảo & Ghi công (References)

Các thí nghiệm đối chứng trong dự án sử dụng tập đặc trưng Silbo Gomero và tham khảo mốc baseline từ công trình:

```bibtex
@inproceedings{LopezGarciaAlfaroContrerasMeyerValeroMas:SPECOM:2025,
  author    = {L{\'o}pez-Garc{\'i}a, Alejandro and Alfaro-Contreras, Mar{\'i}a and Meyer, Julien and Valero-Mas, Jose J.},
  title     = {Whistler Identification in Whistled Spanish (Silbo): A Case Study},
  booktitle = {Proceedings of the 27th International Conference on Speech and Computer (SPECOM)},
  year      = {2025}
}
```
