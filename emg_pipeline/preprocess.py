import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Canonical classes
CLASS_REST = "REST"
CLASS_JAWS = "JAWS"
CLASS_LEFT = "LEFT"
CLASS_RIGHT = "RIGHT"
CLASS_UNKNOWN = "UNKNOWN"

# Default mappings
DEFAULT_LABEL_MAP = {
    "": CLASS_REST,
    "norm": CLASS_REST,
    "rest": CLASS_REST,
    "jaws": CLASS_JAWS,
    "jaw": CLASS_JAWS,
    "jawclench": CLASS_JAWS,
    "jaw clench": CLASS_JAWS,
    "draw": CLASS_JAWS,
    "draws": CLASS_JAWS,
    "click": CLASS_JAWS,
    "clench": CLASS_JAWS,
    "jleft": CLASS_LEFT,
    "j left": CLASS_LEFT,
    "left": CLASS_LEFT,
    "jright": CLASS_RIGHT,
    "j right": CLASS_RIGHT,
    "right": CLASS_RIGHT
}

def clean_and_map_labels(df: pd.DataFrame, label_col: str, label_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Clean labels and map to canonical classes.
    """
    if label_map is None:
        label_map = DEFAULT_LABEL_MAP
        
    def _map_label(val):
        if pd.isna(val):
            return CLASS_REST
        s = str(val).strip().lower()
        if s in label_map:
            return label_map[s]
        # Check partial matches if needed, but dict lookup is safer for now
        # Unknown
        return CLASS_UNKNOWN

    df['canonical_label'] = df[label_col].apply(_map_label)
    
    # Log unknown labels
    unknowns = df[df['canonical_label'] == CLASS_UNKNOWN][label_col].unique()
    if len(unknowns) > 0:
        logger.warning(f"Found unknown labels: {unknowns}. These will be dropped or ignored.")
        
    return df

def get_segments(df: pd.DataFrame, label_col: str = 'canonical_label', fs: float = 250.0) -> List[Dict]:
    """
    Convert sample-level labels into segments.
    Returns a list of dicts: {'label': str, 'start': int, 'end': int, 'duration_ms': float}
    """
    # Run-length encoding
    # Identify changes
    df['change'] = df[label_col].ne(df[label_col].shift()).cumsum()
    
    segments = []
    grouped = df.groupby('change')
    
    for _, group in grouped:
        label = group[label_col].iloc[0]
        start = group.index[0]
        end = group.index[-1] # inclusive
        duration_ms = (end - start + 1) / fs * 1000
        
        segments.append({
            'label': label,
            'start': start,
            'end': end,
            'duration_ms': duration_ms
        })
        
    return segments

def trim_segments(segments: List[Dict], 
                 trim_start_ms: float = 1000, 
                 trim_end_ms: float = 500, 
                 min_rest_ms: float = 2000,
                 fs: float = 250.0) -> List[Dict]:
    """
    Trim start/end of active segments. Filter short rest segments.
    """
    trim_start_samples = int(trim_start_ms / 1000 * fs)
    trim_end_samples = int(trim_end_ms / 1000 * fs)
    
    valid_segments = []
    
    for seg in segments:
        label = seg['label']
        start = seg['start']
        end = seg['end']
        length = end - start + 1
        
        if label == CLASS_UNKNOWN:
            continue
            
        if label == CLASS_REST:
            # Only keep if long enough
            if seg['duration_ms'] >= min_rest_ms:
                valid_segments.append(seg)
        else:
            # Active segment: trim
            new_start = start + trim_start_samples
            new_end = end - trim_end_samples
            
            if new_end > new_start:
                # Update stats
                new_len = new_end - new_start + 1
                new_dur = new_len / fs * 1000
                valid_segments.append({
                    'label': label,
                    'start': new_start,
                    'end': new_end,
                    'duration_ms': new_dur,
                    'orig_start': start,
                    'orig_end': end
                })
            else:
                logger.warning(f"Segment {label} too short to trim (dur={seg['duration_ms']:.1f}ms). Dropped.")
                
    return valid_segments

def analyze_segments(segments: List[Dict], window_ms: float):
    """
    Print stats about segments and validity for windowing.
    """
    logger.info(f"Analyzing {len(segments)} segments for window_ms={window_ms}...")
    counts = {}
    for seg in segments:
        lbl = seg['label']
        counts[lbl] = counts.get(lbl, 0) + 1
        if seg['duration_ms'] < window_ms:
            logger.warning(f"Segment {lbl} duration {seg['duration_ms']:.1f}ms < window {window_ms}ms")
            
    logger.info(f"Segment counts: {counts}")
