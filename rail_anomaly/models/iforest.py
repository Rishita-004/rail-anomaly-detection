"""Isolation Forest on the feature vector (fast baseline, CPU-only)."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IForestModel:
    name = "iforest"

    def __init__(self, n_estimators=300, contamination=0.001, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                                     random_state=random_state, n_jobs=-1)

    def fit(self, X: np.ndarray) -> "IForestModel":
        self.model.fit(self.scaler.fit_transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def raw_score(self, X: np.ndarray) -> np.ndarray:
        """Higher = more anomalous (negated sklearn decision_function)."""
        return -self.model.decision_function(self.transform(X))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "IForestModel":
        return joblib.load(path)
