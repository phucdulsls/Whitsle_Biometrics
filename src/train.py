import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Fix OpenMP duplicate runtime issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import config
from src.audio_io import load_audio, normalize_audio, apply_vad
from src.features import AudioFeatureExtractor

logger = logging.getLogger(__name__)


class BiometricModelWrapper:
    """Encapsulates the trained biometric classification pipeline, classes, and inference methods."""
    
    def __init__(self, pipeline: Pipeline, classes: list[str], backend: str = "mfcc"):
        self.pipeline = pipeline
        self.classes = list(classes)
        self.backend = backend

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """Predicts class and returns posterior probability confidence."""
        if feature_vector.ndim == 1:
            X = feature_vector.reshape(1, -1)
        else:
            X = feature_vector

        probs = self.pipeline.predict_proba(X)[0]
        max_idx = np.argmax(probs)
        pred_label = self.classes[max_idx]
        confidence = float(probs[max_idx])

        prob_dict = {
            self.classes[i]: float(probs[i]) for i in range(len(self.classes))
        }
        return pred_label, confidence, prob_dict

    def save(self, output_path: str):
        """Saves the model wrapper artifact to disk using joblib."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(out))
        print(f"[Train] Model saved successfully to {out}")

    @classmethod
    def load(cls, model_path: str) -> "BiometricModelWrapper":
        """Loads a saved model wrapper artifact from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        wrapper = joblib.load(str(path))
        return wrapper


def train_svm_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    output_path: Optional[str] = None,
    backend: str = "mfcc",
) -> BiometricModelWrapper:
    """Builds and trains a balanced, probability-calibrated SVM pipeline."""
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError(
            f"At least 2 distinct classes are required for classification, found: {unique_classes}"
        )

    print(f"[Train] Training SVM pipeline on {len(X)} samples across classes: {list(unique_classes)}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    probability=True,
                    class_weight="balanced",  # Critical for addressing sample imbalance
                    kernel=config.SVM_KERNEL,
                    C=config.SVM_C,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )

    pipeline.fit(X, y)

    # Cross-validation if sample size allows
    min_class_samples = min(np.sum(y == c) for c in unique_classes)
    n_splits = min(5, min_class_samples)
    if n_splits >= 2:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
        scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1_macro")
        print(f"[Train] {n_splits}-Fold Cross-Validation Macro-F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

    wrapper = BiometricModelWrapper(
        pipeline=pipeline,
        classes=list(pipeline.classes_),
        backend=backend,
    )

    if output_path is not None:
        wrapper.save(output_path)

    return wrapper


def load_dataset_from_directory(
    data_dir: str = str(config.DATA_DIR),
    backend: str = "mfcc",
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads all .wav files from labeled subdirectories in data_dir."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    extractor = AudioFeatureExtractor(backend=backend)
    X, y = [], []

    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise ValueError(f"No class subdirectories found in {data_dir}. Expected data/raw/<label>/*.wav")

    print(f"[Train] Scanning dataset in {data_dir}...")
    for label_dir in subdirs:
        label = label_dir.name
        wav_files = list(label_dir.glob("*.wav"))
        print(f"  Found {len(wav_files)} files for class '{label}'")

        for wav_path in wav_files:
            try:
                audio, sr = load_audio(str(wav_path))
                audio = normalize_audio(audio)
                audio = apply_vad(audio, sr=sr)
                feat = extractor.extract(audio, sr=sr)
                X.append(feat)
                y.append(label)
            except Exception as e:
                logger.warning(f"Error loading {wav_path}: {e}")

    if len(X) == 0:
        raise ValueError(f"No valid audio features extracted from {data_dir}")

    return np.array(X, dtype=np.float32), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train Biometric Identification SVM Model")
    parser.add_argument("--data-dir", type=str, default=str(config.DATA_DIR), help="Path to raw audio dataset")
    parser.add_argument("--output", type=str, default=str(config.MODEL_PATH), help="Path to save model artifact")
    parser.add_argument("--backend", type=str, default="mfcc", choices=["mfcc", "whisper"], help="Feature backend")
    args = parser.parse_args()

    try:
        X, y = load_dataset_from_directory(args.data_dir, backend=args.backend)
        train_svm_pipeline(X, y, output_path=args.output, backend=args.backend)
        print("[Train] Training completed successfully!")
    except Exception as e:
        print(f"[Train Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
