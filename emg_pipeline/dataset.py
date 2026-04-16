import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import GroupKFold
from .preprocess import CLASS_REST

logger = logging.getLogger(__name__)

def create_windows(data: pd.DataFrame, 
                  segments: List[Dict], 
                  fs: float, 
                  window_ms: float, 
                  step_ms: float) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Create windows from segments.
    Returns:
        X_indices: (n_windows, 2) [start, end] indices in original dataframe
        y: (n_windows,) labels
        meta: DataFrame with metadata (segment_id, original_start, etc.)
    """
    window_samples = int(window_ms / 1000 * fs)
    step_samples = int(step_ms / 1000 * fs)
    
    indices_list = []
    labels_list = []
    meta_list = []
    
    for seg_id, seg in enumerate(segments):
        start = int(seg['start'])
        end = int(seg['end'])
        label = seg['label']
        
        # Slide window
        # Ensure we don't go past the end
        curr = start
        while curr + window_samples <= end:
            indices_list.append([curr, curr + window_samples])
            labels_list.append(label)
            meta_list.append({
                'segment_id': seg_id,
                'start_sample': curr,
                'end_sample': curr + window_samples,
                'label': label
            })
            curr += step_samples
            
    return np.array(indices_list), np.array(labels_list), pd.DataFrame(meta_list)

def split_data(X_indices, y, meta, method='segment_split', test_size=0.3):
    """
    Split data into train/test indices.
    Returns: train_idx, test_idx (arrays of indices into X_indices/y)
    """
    n_samples = len(y)
    indices = np.arange(n_samples)
    
    if method == 'segment_split':
        from sklearn.model_selection import StratifiedGroupKFold
        # Leave groups out, but try to balance classes
        groups = meta['segment_id'].values
        # Use a fixed random_state if possible, but StratifiedGroupKFold ensures stratification
        sgkf = StratifiedGroupKFold(n_splits=int(1/test_size), shuffle=True, random_state=42)
        
        # Just take the first split
        # StratifiedGroupKFold.split yields train, test
        train_idx, test_idx = next(sgkf.split(X_indices, y, groups))
        return train_idx, test_idx
        
    elif method == 'time_split':
        # Split chronological: first 70% train, last 30% test
        # But we should do this PER CLASS to maintain balance?
        # Or just strictly time?
        # Implementation plan says: "train first 70%, test last 30% within each class"
        
        train_indices = []
        test_indices = []
        
        # Get unique classes
        classes = np.unique(y)
        for cls in classes:
            cls_indices = indices[y == cls]
            split_point = int(len(cls_indices) * (1 - test_size))
            train_indices.extend(cls_indices[:split_point])
            test_indices.extend(cls_indices[split_point:])
            
        return np.array(train_indices), np.array(test_indices)
        
    else:
        raise ValueError(f"Unknown split method: {method}")

def normalize_features(X_train, X_test, y_train, method='rest_baseline'):
    """
    Normalize features.
    Returns X_train_norm, X_test_norm, scaler (object or params)
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    if method == 'rest_baseline':
        # Fit on REST samples in train
        rest_mask = (y_train == CLASS_REST)
        if np.sum(rest_mask) > 10: # Minimum samples
            logger.info(f"Normalizing using {np.sum(rest_mask)} REST samples.")
            scaler.fit(X_train[rest_mask])
        else:
            logger.warning("Insufficient REST samples for baseline normalization. Fallback to global.")
            scaler.fit(X_train)
    elif method == 'global':
        scaler.fit(X_train)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
        
    X_train_norm = scaler.transform(X_train)
    X_test_norm = scaler.transform(X_test)
    
    return X_train_norm, X_test_norm, scaler
