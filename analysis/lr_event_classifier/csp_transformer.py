from typing import Any

import numpy as np
import scipy.signal
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin


class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, lowcut: float = 8.0, highcut: float = 30.0, fs: float = 250.0, order: int = 2):
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

    def fit(self, X: np.ndarray, y: Any = None) -> "BandpassFilter":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X shape: (n_trials, n_channels, n_times)
        nyq = 0.5 * self.fs
        b, a = scipy.signal.butter(self.order, [self.lowcut / nyq, self.highcut / nyq], btype="band")
        
        # apply filtfilt across the time axis (axis=-1)
        X_filtered = scipy.signal.filtfilt(b, a, X, axis=-1)
        return X_filtered


class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.filters_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPTransformer":
        # X shape: (n_trials, n_channels, n_times)
        # y shape: (n_trials,)
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(unique_classes)}")
            
        c1, c2 = unique_classes[0], unique_classes[1]
        
        def compute_covariance(X_class: np.ndarray) -> np.ndarray:
            _, n_channels, n_times = X_class.shape
            covs = []
            for trial in X_class:
                cov = np.cov(trial)
                trace = np.trace(cov)
                if trace > 0:
                    cov /= trace
                covs.append(cov)
            return np.mean(covs, axis=0)
            
        cov1 = compute_covariance(X[y == c1])
        cov2 = compute_covariance(X[y == c2])
        
        # Solve generalized eigenvalue problem: cov1 * V = lambda * (cov1 + cov2) * V
        evals, evecs = scipy.linalg.eigh(cov1, cov1 + cov2)
        
        # Sort eigenvectors by eigenvalues in descending order
        sort_idx = np.argsort(evals)[::-1]
        evecs = evecs[:, sort_idx]
        
        self.filters_ = evecs.T  # Shape: (n_channels, n_channels)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # X shape: (n_trials, n_channels, n_times)
        if self.filters_ is None:
            raise RuntimeError("CSPTransformer is not fitted.")
            
        n_trials, n_channels, n_times = X.shape
        
        # Extract top (n_components // 2) and bottom (n_components // 2) components
        # Clamp to max channels just in case channel count is smaller than requested components
        half = min(self.n_components // 2, n_channels // 2)
        if half == 0:
            raise ValueError("n_components must be at least 2 and n_channels must be at least 2.")
            
        idx = np.concatenate([np.arange(half), n_channels - np.arange(half) - 1])
        
        W = self.filters_[idx]  # Shape: (len(idx), n_channels)
        
        features = np.zeros((n_trials, len(idx)))
        for i in range(n_trials):
            projected = np.dot(W, X[i])  # Shape: (len(idx), n_times)
            var = np.var(projected, axis=1)
            features[i] = np.log10(np.maximum(var, 1e-12))
            
        return features
