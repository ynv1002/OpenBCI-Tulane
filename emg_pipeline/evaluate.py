import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import logging
from typing import Dict, List
from .preprocess import CLASS_REST

logger = logging.getLogger(__name__)

def get_metrics(y_true, y_pred, classes: List[str]) -> Dict:
    """
    Compute standard classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    # Per-class F1
    # We want it in a dict
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    # We'll return it as a DataFrame
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    
    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "report": report,
        "confusion_matrix": cm_df
    }
    
    return metrics

def calculate_rest_fpr(y_true, y_pred):
    """
    Calculate False Positive Rate for REST class.
    (Percentage of REST windows predicted as non-REST)
    """
    rest_mask = (y_true == CLASS_REST)
    if np.sum(rest_mask) == 0:
        return 0.0
        
    rest_predictions = y_pred[rest_mask]
    false_positives = np.sum(rest_predictions != CLASS_REST)
    fpr = false_positives / len(rest_predictions)
    return fpr

def approximate_detection_latency(y_true, y_pred, meta_df: pd.DataFrame, window_len_samples: int, step_samples: int, fs: float, n_consensus: int = 5):
    """
    Approximate detection latency.
    For each non-REST segment, find time until model consistently predicts non-REST.
    
    Args:
        y_true, y_pred: Arrays of labels.
        meta: Metadata dataframe aligned with y_true/y_pred.
        
    Returns:
        List of latencies in ms.
    """
    # 1. Group by segment
    # We need to rely on segment_id in meta
    if 'segment_id' not in meta_df.columns:
        return []
        
    latencies = []
    
    # Add predictions to meta for easier grouping
    df = meta_df.copy()
    df['y_true'] = y_true
    df['y_pred'] = y_pred
    
    # Filter for ground-truth active segments
    # We iterate unique segment_ids
    for seg_id in df['segment_id'].unique():
        seg_data = df[df['segment_id'] == seg_id]
        true_label = seg_data['y_true'].iloc[0]
        
        if true_label == CLASS_REST:
            continue
            
        # Analyze predictions over time
        # We want the first time we see N consecutive non-REST predictions?
        # Or N consecutive CORRECT predictions?
        # The prompt says: "model's majority vote over last N windows becomes non-REST"
        # Let's simple check: rolling window of N, check if all are non-REST
        # Better: check if MAJORITY is non-REST?
        # Prompt: "majority vote over last N windows becomes non-REST"
        
        preds = seg_data['y_pred'].values
        starts = seg_data['start_sample'].values
        
        detected = False
        detection_time = None
        
        for i in range(len(preds)):
            # Look back N windows (including current)
            if i < n_consensus - 1:
                window = preds[:i+1]
            else:
                window = preds[i-(n_consensus-1):i+1]
                
            # Count non-REST
            non_rest_count = np.sum(window != CLASS_REST)
            
            if non_rest_count > len(window) / 2:
                # Majority is non-REST
                detected = True
                # The time this decision is made is the END of the current window
                # Latency = (Time of decision) - (Start of segment)
                # Ensure we have segment start
                # meta_df has 'start_sample' of the window
                # We need segment start. We can infer from the first window's start?
                # Actually, the segment might have been trimmed.
                # 'start_sample' is relative to the recording.
                # The segment start in the recording is essentially `seg_data['start_sample'].iloc[0]`
                # because we windowed the TRIMMED segment.
                # So latency is relative to the *trimmed* start.
                
                decision_sample = seg_data['end_sample'].iloc[i]
                segment_start_sample = seg_data['start_sample'].iloc[0]
                
                latency_samples = decision_sample - segment_start_sample
                detection_time = latency_samples / fs * 1000
                break
                
        if detected:
            latencies.append(detection_time)
        else:
            # Missed event
            pass
            
    return latencies
