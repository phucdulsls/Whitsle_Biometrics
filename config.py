import os
from pathlib import Path

# Fix OpenMP duplicate runtime issue on Windows Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
DOCS_DIR = BASE_DIR / "docs"
TESTS_DIR = BASE_DIR / "tests"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Audio Parameters
SAMPLE_RATE = 16000          # 16 kHz: Standard for speech & whistling biometrics
DURATION = 3.0               # Default recording duration in seconds
CHANNELS = 1                 # Mono
DTYPE = "float32"

# Acoustic Frequency Range (Covers low voice 85Hz up to high whistle harmonics ~4000-8000Hz)
FMIN = 50.0
FMAX = 8000.0
N_MFCC = 20                  # Base MFCC coefficients
N_FFT = 2048
HOP_LENGTH = 512

# Voice Activity Detection (VAD) & Normalization Parameters
VAD_FRAME_LENGTH = 2048
VAD_HOP_LENGTH = 512
VAD_TOP_DB = 25.0            # Decibels below peak to consider silence
PEAK_TARGET = 0.95           # Peak normalization ceiling

# Feature Dimension:
# MFCC (20) + Delta (20) + Delta-Delta (20) = 60
# Mean pooling (60) + Std pooling (60) = 120 dimensions
FEATURE_DIM_MFCC = 120

# Biometric Classification Parameters
CONFIDENCE_THRESHOLD = 0.70  # Minimum posterior probability to confirm identity
RANDOM_STATE = 42
SVM_KERNEL = "rbf"
SVM_C = 10.0

