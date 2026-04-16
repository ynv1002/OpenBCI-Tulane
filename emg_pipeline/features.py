import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def rms(x):
    return np.sqrt(np.mean(x**2))

def mav(x):
    return np.mean(np.abs(x))

def variance(x):
    return np.var(x)

def waveform_length(x):
    return np.sum(np.abs(np.diff(x)))

def zero_crossing(x, threshold=0):
    # Number of times signal crosses zero
    # Use a small threshold to avoid noise
    x_c = x - np.mean(x)
    signs = np.sign(x_c)
    diffs = np.diff(signs)
    count = np.sum(np.abs(diffs) > 0)
    return count

def slope_sign_changes(x, threshold=0):
    # Number of times slope changes sign
    diffs = np.diff(x)
    signs = np.sign(diffs)
    changes = np.diff(signs)
    return np.sum(np.abs(changes) > 0)

# Optional frequency features
try:
    from scipy.signal import welch
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("Scipy not found. Frequency domain features will be skipped.")

def mean_freq(f, Pxx):
    sum_Pxx = np.sum(Pxx)
    if sum_Pxx == 0:
        return 0.0
    return np.sum(f * Pxx) / sum_Pxx

def median_freq(f, Pxx):
    cum_power = np.cumsum(Pxx)
    if len(cum_power) == 0:
        return 0.0
    total_power = cum_power[-1]
    if total_power == 0:
        return 0.0
    idx = np.searchsorted(cum_power, total_power / 2)
    return f[idx]

def bandpower(f, Pxx, fmin, fmax):
    idx = np.logical_and(f >= fmin, f <= fmax)
    return np.trapz(Pxx[idx], f[idx])


def extract_features(window_data: np.ndarray, fs: float = 250.0) -> np.ndarray:
    """
    Extract features from a window.
    window_data: shape (n_samples, n_channels)
    
    Returns: flattened feature vector
    """
    n_samples, n_channels = window_data.shape
    features = []
    
    # Time domain
    for ch in range(n_channels):
        x = window_data[:, ch]
        features.append(rms(x)) 
        features.append(mav(x))
        features.append(variance(x))
        features.append(waveform_length(x))
        features.append(zero_crossing(x))
        features.append(slope_sign_changes(x))

        # Frequency domain
        if SCIPY_AVAILABLE:
            f, Pxx = welch(x, fs=fs, nperseg=n_samples) # Use entire window
            # Handle potential NaNs in Pxx or empty windows
            Pxx = np.nan_to_num(Pxx)
            
            features.append(mean_freq(f, Pxx))
            features.append(median_freq(f, Pxx))
            # Check for NaN in bandpower
            bp1 = bandpower(f, Pxx, 20, 60)
            bp2 = bandpower(f, Pxx, 60, 120)
            features.append(np.nan_to_num(bp1))
            features.append(np.nan_to_num(bp2))
            
    return np.nan_to_num(np.array(features))

def get_feature_names(n_channels: int) -> List[str]:
    names = []
    base_feats = ["RMS", "MAV", "VAR", "WL", "ZC", "SSC"]
    if SCIPY_AVAILABLE:
        base_feats.extend(["MNF", "MDF", "BP_20_60", "BP_60_120"])
        
    for ch in range(n_channels):
        for feat in base_feats:
            names.append(f"Ch{ch}_{feat}")
            
    return names
