import os
from pathlib import Path

# OpenMP runtime fix for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "saved_models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Feature Files
FEAT_MFCC_DELTADELTA = FEATURES_DIR / "mfcc_deltadelta_20.csv"
FEAT_MFCC_MEAN = FEATURES_DIR / "mfcc_mean_20.csv"
FEAT_WHISPER_TINY = FEATURES_DIR / "whisper_tiny.csv"
FEAT_WHISPER_MEDIUM = FEATURES_DIR / "whisper_medium.csv"
FEAT_WHISPER_SMALL = FEATURES_DIR / "whisper_small.csv"
FEAT_WAV2VEC_1024 = FEATURES_DIR / "wav2vec_1024.csv"

# Evaluation Protocol
N_SPLITS = 5
RANDOM_STATE = 42

# Confidence & Thresholds
CONFIDENCE_THRESHOLD = 0.70
