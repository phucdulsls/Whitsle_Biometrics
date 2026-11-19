import os
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

# OpenMP fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from improvement_01 import config


def parse_feature_row(feat_str: str) -> np.ndarray:
    """Parses stringified numpy array from features CSV."""
    cleaned = feat_str.replace("[", "").replace("]", "").replace(",", "")
    return np.fromstring(cleaned, sep=" ", dtype=np.float32)


def load_single_feature(feature_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a single feature dataset from CSV.
    
    Returns:
        X: numpy array of shape (N, D)
        y: numpy array of labels of shape (N,)
    """
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found at: {feature_path}")

    df = pd.read_csv(feature_path)
    X = np.stack([parse_feature_row(s) for s in df["features"]])
    y = df["label"].values
    return X, y


def load_all_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Loads both baseline single-feature datasets and improved fused datasets.
    
    Returns:
        Dictionary mapping dataset configuration name to (X, y) tuple.
    """
    print("[DataLoader] Loading underlying feature matrices...")
    X_mfcc_dd, y = load_single_feature(config.FEAT_MFCC_DELTADELTA)
    X_w_med, _ = load_single_feature(config.FEAT_WHISPER_MEDIUM)
    X_w_tiny, _ = load_single_feature(config.FEAT_WHISPER_TINY)

    print(f"  - MFCC Delta-Delta Shape : {X_mfcc_dd.shape}")
    print(f"  - Whisper Medium Shape   : {X_w_med.shape}")
    print(f"  - Whisper Tiny Shape     : {X_w_tiny.shape}")

    # 1. Hybrid Fusion: MFCC (Frequency dynamics) + Whisper Medium (Deep Semantic)
    X_hybrid = np.hstack([X_mfcc_dd, X_w_med])

    # 2. Multi-Scale Fusion: MFCC + Whisper Medium + Whisper Tiny (Multi-resolution Audio Embeddings)
    X_multiscale = np.hstack([X_mfcc_dd, X_w_med, X_w_tiny])

    print(f"  - Fused Hybrid Shape     : {X_hybrid.shape}")
    print(f"  - Fused MultiScale Shape : {X_multiscale.shape}")

    return {
        "Baseline_MFCC_DeltaDelta": (X_mfcc_dd, y),
        "Baseline_Whisper_Tiny": (X_w_tiny, y),
        "Baseline_Whisper_Medium": (X_w_med, y),
        "Improved_Hybrid_MFCC_WhisperMed": (X_hybrid, y),
        "Improved_MultiScale_Whisper_MFCC": (X_multiscale, y),
    }
