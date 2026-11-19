import os
from pathlib import Path

# OpenMP runtime fix for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FEATURES_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = BASE_DIR / "results"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Datasets
FEAT_MFCC_DELTADELTA = FEATURES_DIR / "mfcc_deltadelta_20.csv"
FEAT_WHISPER_TINY = FEATURES_DIR / "whisper_tiny.csv"
FEAT_WHISPER_BASE = FEATURES_DIR / "whisper_base.csv"
FEAT_WHISPER_SMALL = FEATURES_DIR / "whisper_small.csv"
FEAT_WHISPER_MEDIUM = FEATURES_DIR / "whisper_medium.csv"
FEAT_WAV2VEC_1024 = FEATURES_DIR / "wav2vec_1024.csv"
FEAT_WAV2VEC_256 = FEATURES_DIR / "wav2vec_256.csv"

# Evaluation Protocol
N_SPLITS = 5
RANDOM_STATE = 42

# LDA & Cosine Parameters
LDA_SOLVER = "eigen"
LDA_SHRINKAGE = "auto"   # Ledoit-Wolf analytical shrinkage for singular covariance
COSINE_TEMPERATURE = 10.0 # Softmax temperature scaling for cosine similarities
