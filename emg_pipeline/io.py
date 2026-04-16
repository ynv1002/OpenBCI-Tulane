import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load EMG data from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded data from {csv_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {csv_path}: {e}")
        raise

def get_label_column(df: pd.DataFrame) -> str:
    """
    Identify the label column in the DataFrame.
    Prioritizes 'label' (case-insensitive), otherwise uses the last column.
    
    Args:
        df (pd.DataFrame): The data.
        
    Returns:
        str: Name of the label column.
    """
    # Check for 'label' case-insensitive
    for col in df.columns:
        if col.lower().strip() == 'label':
            logger.info(f"Found label column: {col}")
            return col
            
    # Fallback to last column
    last_col = df.columns[-1]
    logger.warning(f"Label column not found by name. parsing last column '{last_col}' as label.")
    return last_col

def get_feature_columns(df: pd.DataFrame, label_col: str) -> list:
    """
    Identify feature columns (EMG channels).
    Exclude label, timestamp, index, and other metadata.
    """
    exclude = {label_col, 'Sample Index', 'Timestamp', 'Timestamp (Formatted)', 'ISO 8601', 'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2', 'Other'}
    
    feature_cols = []
    for col in df.columns:
        if col not in exclude and 'accel' not in col.lower() and 'timestamp' not in col.lower() and 'index' not in col.lower():
            feature_cols.append(col)
            
    logger.info(f"Identified {len(feature_cols)} feature columns: {feature_cols}")
    return feature_cols

