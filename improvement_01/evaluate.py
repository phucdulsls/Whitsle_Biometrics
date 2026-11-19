import os
from typing import Dict, Any, List

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from improvement_01 import config


def evaluate_cv(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = config.N_SPLITS,
    random_state: int = config.RANDOM_STATE,
) -> Dict[str, float]:
    """Performs Stratified K-Fold Cross Validation and computes comprehensive metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    macro_f1_list = []
    weighted_f1_list = []
    acc_list = []
    precision_list = []
    recall_list = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)

        macro_f1_list.append(macro_f1)
        weighted_f1_list.append(weighted_f1)
        acc_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)

    return {
        "macro_f1_mean": float(np.mean(macro_f1_list)),
        "macro_f1_std": float(np.std(macro_f1_list)),
        "accuracy_mean": float(np.mean(acc_list)),
        "accuracy_std": float(np.std(acc_list)),
        "weighted_f1_mean": float(np.mean(weighted_f1_list)),
        "weighted_f1_std": float(np.std(weighted_f1_list)),
        "precision_mean": float(np.mean(precision_list)),
        "recall_mean": float(np.mean(recall_list)),
    }
