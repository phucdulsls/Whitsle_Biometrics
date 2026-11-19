import os
from typing import Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from improvement_01 import config


class SampleWiseMinMaxTransformer(BaseEstimator, TransformerMixin):
    """Replicates the baseline paper's sample-wise (axis=1) normalization."""
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_min = X.min(axis=1, keepdims=True)
        X_max = X.max(axis=1, keepdims=True)
        denom = np.where(X_max - X_min == 0, 1.0, X_max - X_min)
        return (X - X_min) / denom


def build_paper_baseline_model(classifier_type: str = "svc") -> Pipeline:
    """Builds the exact pipeline configuration used in the SPECOM 2025 paper."""
    if classifier_type == "svc":
        clf = SVC(kernel="rbf", random_state=config.RANDOM_STATE)
    elif classifier_type == "rf":
        clf = RandomForestClassifier(n_estimators=500, random_state=config.RANDOM_STATE)
    elif classifier_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=3)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return Pipeline([
        ("minmax_sample", SampleWiseMinMaxTransformer()),
        ("clf", clf)
    ])


def build_improved_pipeline(classifier_type: str = "svc") -> Pipeline:
    """Builds improved pipeline with Feature-wise Standard Scaling + L2 Normalization (Cosine geometry)."""
    if classifier_type == "svc":
        clf = SVC(
            C=10.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=config.RANDOM_STATE,
        )
    elif classifier_type == "lr":
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=config.RANDOM_STATE,
        )
    elif classifier_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=1000,
            random_state=config.RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("l2_norm", Normalizer(norm="l2")),
        ("clf", clf)
    ])


def build_ensemble_voting_model() -> VotingClassifier:
    """Builds an Ensemble Soft-Voting Classifier combining calibrated SVM, Regularized LR, and MLP.
    
    This architecture achieves maximum robustness on imbalanced whistler classes by fusing:
    - Support Vector boundary separation with balanced class penalty.
    - Logistic Regression linear hyperplane margin.
    - Non-linear representation learning via Multi-Layer Perceptron.
    """
    pipe_svc = build_improved_pipeline("svc")
    pipe_lr = build_improved_pipeline("lr")
    pipe_mlp = build_improved_pipeline("mlp")

    ensemble = VotingClassifier(
        estimators=[
            ("svc", pipe_svc),
            ("lr", pipe_lr),
            ("mlp", pipe_mlp),
        ],
        voting="soft",
        weights=[2.0, 1.5, 1.0],  # Give higher weight to calibrated SVM and LR
    )
    return ensemble
