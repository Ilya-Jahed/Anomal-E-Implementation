"""
anomaly_detector.py
====================

Final stage of the Anomal-E pipeline (Section 4.4 / Algorithm 2, lines 11-14).

    Raw NetFlow CSV -> preprocessor.py -> graph_builder.py -> AnomalESAGEEncoder
    -> 256-dim edge embeddings -> AnomalEDetector (THIS FILE) -> anomaly score / label

Wraps four classical unsupervised anomaly detectors from PyOD (PCA, HBOS, CBLOF,
Isolation Forest) behind one consistent fit/predict/score API. None of them ever
see attack labels -- see README.md for full details, math, and paper references.
"""

import numpy as np
import torch
from pyod.models.pca import PCA
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest


class AnomalEDetector:
    """
    Unified wrapper around PyOD's unsupervised anomaly detectors: 'pca', 'hbos',
    'cblof', 'iforest' (paper refs [7], [9], [8], [6] respectively). Takes the
    edge embeddings produced by the E-GraphSAGE encoder and scores/flags each
    flow as benign or anomalous, without ever training on labels.

    See README.md for what each algorithm does internally and how `contamination`
    maps to the paper's Table 2 grid search.

    Usage:
        detector = AnomalEDetector(model_name='iforest', contamination=0.04)
        detector.fit(z_train)
        labels = detector.predict(z_test)             # 0 = benign, 1 = attack
        scores = detector.get_anomaly_scores(z_test)   # continuous severity
    """

    def __init__(self, model_name='pca', contamination=0.1, random_state=42, **kwargs):
        """
        Args:
            model_name (str): 'pca', 'hbos', 'cblof', or 'iforest' (case-insensitive).
            contamination (float): expected anomaly fraction in (0, 0.5), used by
                PyOD to set the decision threshold. No labels required -- see README.
            random_state (int): reproducibility seed (not accepted by HBOS, which
                is deterministic).
            **kwargs: extra PyOD hyperparameters (n_components, n_bins, n_clusters,
                n_estimators, etc.).

        Raises:
            ValueError: if model_name isn't one of the four supported options.
        """
        self.model_name = model_name.lower()
        self.contamination = contamination

        if self.model_name == 'pca':
            self.model = PCA(contamination=contamination, random_state=random_state, **kwargs)
        elif self.model_name == 'hbos':
            self.model = HBOS(contamination=contamination, **kwargs)  # no random_state arg
        elif self.model_name == 'cblof':
            self.model = CBLOF(contamination=contamination, random_state=random_state, **kwargs)
        elif self.model_name == 'iforest':
            self.model = IForest(contamination=contamination, random_state=random_state, **kwargs)
        else:
            raise ValueError(f"[ERROR] Model '{model_name}' is not supported. Choose from: pca, hbos, cblof, iforest")

    def _prepare_data(self, X):
        """Convert torch.Tensor (detach -> cpu -> numpy) or array-like input into a plain np.ndarray."""
        if isinstance(X, torch.Tensor):
            return X.detach().cpu().numpy()
        return np.array(X)

    def fit(self, X):
        """
        Train the detector on embeddings X (n_samples, n_features), unsupervised.

        Returns:
            AnomalEDetector: self, for chaining.
        """
        X_np = self._prepare_data(X)
        self.model.fit(X_np)
        return self

    def predict(self, X):
        """
        Return hard 0/1 labels for X using the contamination-derived threshold.

        Returns:
            np.ndarray: shape (n_samples,), 0 = benign, 1 = anomaly.
        """
        X_np = self._prepare_data(X)
        return self.model.predict(X_np)

    def get_anomaly_scores(self, X):
        """
        Return continuous anomaly scores for X (higher = more anomalous).

        Returns:
            np.ndarray: shape (n_samples,), float scores.
        """
        X_np = self._prepare_data(X)
        return self.model.decision_function(X_np)