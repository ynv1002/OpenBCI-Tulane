import argparse
import os
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from . import io, preprocess, dataset, features, models, evaluate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(args):
    # 1. Load Data
    logger.info(f"Loading data from {args.csv}...")
    df = io.load_data(args.csv)
    
    # 2. Identify Columns
    label_col = io.get_label_column(df)
    feature_cols = io.get_feature_columns(df, label_col)
    
    # 3. Preprocess
    logger.info("Preprocessing...")
    # Optional mapping from args? For now stick to default
    df = preprocess.clean_and_map_labels(df, label_col)
    
    # Get segments
    segments = preprocess.get_segments(df, label_col='canonical_label', fs=args.fs)
    logger.info(f"Found {len(segments)} segments initially.")
    
    # Trim segments
    valid_segments = preprocess.trim_segments(segments, 
                                            trim_start_ms=args.trim_start_ms, 
                                            trim_end_ms=args.trim_end_ms, 
                                            min_rest_ms=args.min_rest_ms,
                                            fs=args.fs)
    logger.info(f"Retained {len(valid_segments)} valid segments after trimming.")
    
    # 4. Loop over window sizes
    window_sizes = args.window_ms
    results_summary = []
    
    # Prepare output dir
    os.makedirs(args.outdir, exist_ok=True)
    
    for win_ms in window_sizes:
        logger.info(f"--- Processing window size: {win_ms} ms ---")
        
        # Check strictness
        preprocess.analyze_segments(valid_segments, win_ms)
        
        # Windowing
        X_indices, y, meta = dataset.create_windows(df, valid_segments, args.fs, win_ms, args.step_ms)
        logger.info(f"Created {len(y)} windows.")
        
        if len(y) == 0:
            logger.warning("No windows created! Skipping.")
            continue
            
        # Feature Extraction
        logger.info("Extracting features...")
        X_features = []
        # Optimization: retrieve numeric data once
        data_matrix = df[feature_cols].values
        
        for start, end in X_indices:
            window_data = data_matrix[start:end, :]
            feat_vec = features.extract_features(window_data, fs=args.fs)
            X_features.append(feat_vec)
            
        X = np.array(X_features)
        
        # Save feature names
        feat_names = features.get_feature_names(len(feature_cols))
        
        # Save processed data
        # Only save if requested or for debugging? 
        # Prompt: "Save for each window size: a DataFrame... and optionally npz"
        
        # 5. Split
        logger.info(f"Splitting data using {args.split}...")
        train_idx, test_idx = dataset.split_data(X_indices, y, meta, method=args.split, test_size=0.3)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        meta_test = meta.iloc[test_idx]
        
        # 6. Normalize
        logger.info("Normalizing...")
        X_train_norm, X_test_norm, scaler = dataset.normalize_features(X_train, X_test, y_train, method='rest_baseline')
        
        # 7. Models & Evaluation
        models_to_train = ['lda', 'lr', 'svm']
        
        for model_name in models_to_train:
            logger.info(f"Training {model_name}...")
            clf = models.EMGModel(model_type=model_name)
            clf.fit(X_train_norm, y_train)
            
            y_pred = clf.predict(X_test_norm)
            
            # Metrics
            classes = [preprocess.CLASS_REST, preprocess.CLASS_JAWS, preprocess.CLASS_LEFT, preprocess.CLASS_RIGHT]
            metrics = evaluate.get_metrics(y_test, y_pred, classes)
            
            # Additional metrics
            rest_fpr = evaluate.calculate_rest_fpr(y_test, y_pred)
            latencies = evaluate.approximate_detection_latency(y_test, y_pred, meta_test, 
                                                             window_len_samples=int(win_ms/1000*args.fs), 
                                                             step_samples=int(args.step_ms/1000*args.fs), 
                                                             fs=args.fs)
            avg_latency = np.mean(latencies) if latencies else float('nan')
            
            # Store results
            res_entry = {
                "window_ms": win_ms,
                "model": model_name,
                "accuracy": metrics['accuracy'],
                "f1_macro": metrics['f1_macro'],
                "rest_fpr": rest_fpr,
                "avg_latency_ms": avg_latency
            }
            results_summary.append(res_entry)
            
            # Save confusion matrix
            cm_filename = f"confusion_matrix_{win_ms}ms_{model_name}.csv"
            metrics['confusion_matrix'].to_csv(os.path.join(args.outdir, cm_filename))
            
    # Save summary
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(args.outdir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Pipeline complete. Results saved to {args.outdir}")
    print(summary_df)

def main():
    parser = argparse.ArgumentParser(description="EMG Offline Pipeline")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--fs", type=float, default=250.0, help="Sampling rate (Hz)")
    parser.add_argument("--subject_id", type=str, default="S1", help="Subject ID")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--window_ms", nargs="+", type=int, default=[150, 200, 250], help="Window sizes in ms")
    parser.add_argument("--step_ms", type=float, default=50.0, help="Step size in ms")
    parser.add_argument("--trim_start_ms", type=float, default=1000.0, help="Trim start of segments (ms)")
    parser.add_argument("--trim_end_ms", type=float, default=500.0, help="Trim end of segments (ms)")
    parser.add_argument("--min_rest_ms", type=float, default=2000.0, help="Minimum duration for REST segments (ms)")
    parser.add_argument("--split", default="segment_split", choices=["segment_split", "time_split"], help="Split method")
    
    args = parser.parse_args()
    run_pipeline(args)

if __name__ == "__main__":
    main()
