from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scipy.signal
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.lr_event_classifier.cross_session_evaluator import evaluate_models
from analysis.lr_event_classifier.event_matrix_builder import (
    LABELS,
    LEFT,
    RIGHT,
    build_event_matrix,
)
from analysis.stepwise_protocol_registry import (
    eeg_lr_main_contracts,
    eeg_lr_stress_contracts,
    lrj_main_contracts,
)
from analysis.utils import (
    PROJECT_ROOT,
    dataframe_to_markdown,
    ensure_output_dir,
    write_markdown,
)

OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "outputs"
TIME_DOMAIN_SUMMARY_PATH = OUTPUT_ROOT / "left_right_coverage_summary.json"
REPORT_PATH = OUTPUT_ROOT / "spectral_results.md"

def extract_bandpower(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    idx = np.logical_and(freqs >= fmin, freqs <= fmax)
    # Numerical integration using the frequency resolution (df)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sum(psd[idx]) * df)


def _spectral_features_v1(window: np.ndarray, channel_columns: list[str]) -> dict[str, float]:
    features = {}
    fs = 250.0
    nperseg = int(fs)
    if len(window) < nperseg:
        nperseg = len(window)
        
    for idx, channel in enumerate(channel_columns):
        sig = window[:, idx]
        freqs, psd = scipy.signal.welch(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        
        # Calculate raw bandpowers
        mu_bp = extract_bandpower(psd, freqs, 8, 12)
        lowbeta_bp = extract_bandpower(psd, freqs, 13, 20)
        beta_bp = extract_bandpower(psd, freqs, 20, 30)
        
        # log10 bandpowers
        features[f"{channel}_mu_logbp"] = float(np.log10(max(mu_bp, 1e-12)))
        features[f"{channel}_lowbeta_logbp"] = float(np.log10(max(lowbeta_bp, 1e-12)))
        features[f"{channel}_beta_logbp"] = float(np.log10(max(beta_bp, 1e-12)))
        
    return features


def _spectral_features_v2(window: np.ndarray, channel_columns: list[str]) -> dict[str, float]:
    features = _spectral_features_v1(window, channel_columns)
    
    n = len(channel_columns)
    for i in range(n):
        for j in range(i + 1, n):
            c1 = channel_columns[i]
            c2 = channel_columns[j]
            
            for band in ["mu", "lowbeta", "beta"]:
                val1 = features[f"{c1}_{band}_logbp"]
                val2 = features[f"{c2}_{band}_logbp"]
                features[f"{c1}_minus_{c2}_{band}_logbp"] = float(val1 - val2)
                
    return features


def _model_bank() -> dict[str, Any]:
    return {
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]
        ),
    }

def run_classifier(
    output_dir: Path,
    include_files: list[str],
    window_sec: float,
    feature_extractor,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)

    event_matrix_df, feature_columns, common_channels = build_event_matrix(
        feature_extractor=feature_extractor,
        include_files=include_files,
        window_sec=window_sec,
    )
    event_matrix_df.to_csv(output_dir / "event_feature_table.csv", index=False)

    eval_outputs = evaluate_models(
        event_matrix_df=event_matrix_df,
        feature_columns=feature_columns,
        model_bank=_model_bank(),
        labels=LABELS,
        output_dir=output_dir,
    )

    summary = eval_outputs["summary"]
    summary["files_used"] = sorted(event_matrix_df["file"].unique().tolist())
    summary["selected_channels"] = common_channels
    summary["feature_count"] = len(feature_columns)
    summary["sample_counts"] = {
        "total": int(len(event_matrix_df)),
        LEFT: int((event_matrix_df["label"] == LEFT).sum()),
        RIGHT: int((event_matrix_df["label"] == RIGHT).sum()),
    }
    return eval_outputs | {"summary": summary}


def _summary_row(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    # Default feature count to string to handle time_domain NA if not injected
    feat_count = summary.get("feature_count")
    feat_count_str = str(feat_count) if feat_count is not None else "N/A"
    return {
        "evaluation_set": label,
        "feature_count": feat_count_str,
        "event_count": int(summary["sample_counts"]["total"]),
        "best_model": str(summary["best_model"]["name"]),
        "pooled_accuracy": float(summary["best_model"]["accuracy"]),
        "pooled_macro_f1": float(summary["best_model"]["macro_f1"]),
    }

def main() -> None:
    ensure_output_dir(OUTPUT_ROOT)
    
    main_lr_files = [contract.filename for contract in eeg_lr_main_contracts()]
    all_lr_files = main_lr_files + [contract.filename for contract in eeg_lr_stress_contracts()]
    combo_lr_lrj_files = all_lr_files + [contract.filename for contract in lrj_main_contracts()]
    
    td_combo_summary = {}
    if TIME_DOMAIN_SUMMARY_PATH.exists():
        time_domain_data = json.loads(TIME_DOMAIN_SUMMARY_PATH.read_text(encoding="utf-8"))
        td_combo_summary = time_domain_data.get("combo_lr_lrj_summary", {})
    
    td_row = {}
    if td_combo_summary:
        td_row = _summary_row("time_domain_baseline (combo)", td_combo_summary)
        td_row["feature_count"] = "32" # Known from previous math extraction
    
    results_v1_1_0 = run_classifier(
        output_dir=OUTPUT_ROOT / "spectral_v1_1.0s",
        include_files=combo_lr_lrj_files,
        window_sec=1.0,
        feature_extractor=_spectral_features_v1,
    )
    v1_1_0_summary = results_v1_1_0["summary"]
    row_v1_1_0 = _summary_row("spectral_v1_1.0s", v1_1_0_summary)

    results_v1_1_5 = run_classifier(
        output_dir=OUTPUT_ROOT / "spectral_v1_1.5s",
        include_files=combo_lr_lrj_files,
        window_sec=1.5,
        feature_extractor=_spectral_features_v1,
    )
    v1_1_5_summary = results_v1_1_5["summary"]
    row_v1_1_5 = _summary_row("spectral_v1_1.5s", v1_1_5_summary)
    
    f1_1_0 = v1_1_0_summary["best_model"]["macro_f1"]
    f1_1_5 = v1_1_5_summary["best_model"]["macro_f1"]
    best_window = 1.0 if f1_1_0 >= f1_1_5 else 1.5
    
    results_v2 = run_classifier(
        output_dir=OUTPUT_ROOT / "spectral_v2_best",
        include_files=combo_lr_lrj_files,
        window_sec=best_window,
        feature_extractor=_spectral_features_v2,
    )
    v2_summary = results_v2["summary"]
    row_v2 = _summary_row(f"spectral_v2_{best_window}s_asym", v2_summary)

    rows = []
    if td_row:
        rows.append(td_row)
    rows.extend([row_v1_1_0, row_v1_1_5, row_v2])
    
    scorecard_df = pd.DataFrame(rows)
    # Format metrics
    for col in ("pooled_accuracy", "pooled_macro_f1"):
        scorecard_df[col] = scorecard_df[col].map(lambda value: f"{float(value):.3f}")

    print("Spectral Motor-Imagery Baseline Review\n")
    print(dataframe_to_markdown(scorecard_df))
    
    lines = [
        "# Spectral Motor-Imagery Baseline Review",
        "",
        "## Feature Dimension Guardrails",
        f"- Common Channels Active: `{', '.join(v2_summary['selected_channels'])}`",
        f"- Spectral v1 feature count: `{row_v1_1_0['feature_count']}` (Base Bandpowers)",
        f"- Spectral v2 feature count: `{row_v2['feature_count']}` (Base + Pairwise Differences)",
        "",
        "## Spectral vs Time-Domain Results",
        dataframe_to_markdown(scorecard_df),
        "",
        "### Interpretation",
        "If `spectral_v1` outperforms the time domain baseline, the core motor imagery signal geometry is highly linearly separable.",
        "If `spectral_v2` provides a major jump, relative left-right structure is the dominant directional component.",
    ]
    write_markdown(REPORT_PATH, "\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
