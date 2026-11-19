import os
from typing import Dict, Tuple, Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, normalize

# OpenMP fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from improvement_02_b1_lda_cosine import config


class LDACosineClassifier(BaseEstimator, ClassifierMixin):
    """Linear Discriminant Analysis Subspace Projection with Length-Normalized Cosine Centroid Scoring.
    
    Architecture (Phương án B1):
    1. StandardScaler: Standardizes features across samples.
    2. Regularized LDA: Projects embeddings into a discriminative (C - 1) subspace maximizing 
       between-class variance while minimizing within-class variance with Ledoit-Wolf shrinkage.
    3. Length Normalization: L2 unit normalization maps vectors onto the unit hypersphere.
    4. Enrolled Centroids: Speaker prototype vectors mu_c = normalize(mean(z_i in class c)).
    5. Cosine Scoring: score(z, c) = cos(z, mu_c) = z . mu_c.
    6. Softmax Calibration: P(c|z) = exp(tau * score) / sum_k exp(tau * score).
    """

    def __init__(
        self,
        solver: str = config.LDA_SOLVER,
        shrinkage: str = config.LDA_SHRINKAGE,
        temperature: float = config.COSINE_TEMPERATURE,
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.temperature = temperature
        
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis(solver=self.solver, shrinkage=self.shrinkage)
        self.centroids_: Dict[Any, np.ndarray] = {}
        self.classes_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDACosineClassifier":
        """Fits Scaler, Regularized LDA, and computes speaker centroids in LDA space."""
        self.classes_ = np.unique(y)
        
        # 1. Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        # 2. Fit Regularized LDA & Transform
        X_lda = self.lda.fit(X_scaled, y).transform(X_scaled)
        
        # 3. L2 Length Normalization (Cosine geometry)
        X_norm = normalize(X_lda, norm="l2")
        
        # 4. Compute Normalized Enrolled Centroids for each speaker/whistler class
        self.centroids_ = {}
        for c in self.classes_:
            class_vecs = X_norm[y == c]
            mean_vec = np.mean(class_vecs, axis=0, keepdims=True)
            self.centroids_[c] = normalize(mean_vec, norm="l2")[0]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects input features into the L2-normalized LDA subspace."""
        X_scaled = self.scaler.transform(X)
        X_lda = self.lda.transform(X_scaled)
        return normalize(X_lda, norm="l2")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Computes pairwise cosine similarity scores between each sample and all class centroids."""
        X_norm = self.transform(X)
        scores = np.zeros((len(X), len(self.classes_)), dtype=np.float32)
        
        for idx, c in enumerate(self.classes_):
            scores[:, idx] = np.dot(X_norm, self.centroids_[c])
            
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculates calibrated class probabilities using temperature-scaled softmax on cosine similarities."""
        scores = self.decision_function(X) * self.temperature
        # Numerically stable softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the identity with highest cosine similarity in the LDA subspace."""
        scores = self.decision_function(X)
        best_indices = np.argmax(scores, axis=1)
        return self.classes_[best_indices]
