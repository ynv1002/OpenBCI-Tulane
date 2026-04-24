from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.lr_event_classifier.cross_session_evaluator import evaluate_tensor_models
from analysis.lr_event_classifier.event_matrix_builder import (
    LABELS,
    LEFT,
    RIGHT,
    build_tensor_matrix,
)
from analysis.lr_event_classifier.csp_transformer import BandpassFilter, CSPTransformer
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
REPORT_PATH = OUTPUT_ROOT / "csp_results.md"

def _model_bank(n_components: int = 4) -> dict[str, Any]:
    return {
        "LDA": Pipeline([
            ("bandpass", BandpassFilter(lowcut=8.0, highcut=30.0, fs=250.0)),
            ("csp", CSPTransformer(n_components=n_components)),
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis())
        ]),
        "LogisticRegression": Pipeline([
            ("bandpass", BandpassFilter(lowcut=8.0, highcut=30.0, fs=250.0)),
            ("csp", CSPTransformer(n_components=n_components)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))
        ]),
    }

def run_classifier(
    output_dir: Path,
    include_files: list[str],
    window_sec: float,
    n_components: int,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)

    X_tensor, metadata_df, common_channels = build_tensor_matrix(
        include_files=include_files,
        window_sec=window_sec,
    )
    # We do not save an event_feature_table.csv because we are returning tensors, not flattened tabular features.

    eval_outputs = evaluate_tensor_models(
        X_tensor=X_tensor,
        metadata_df=metadata_df,
        model_bank=_model_bank(n_components=n_components),
        labels=LABELS,
        output_dir=output_dir,
    )

    summary = eval_outputs["summary"]
    summary["files_used"] = sorted(metadata_df["file"].unique().tolist())
    summary["selected_channels"] = common_channels
    summary["feature_count"] = min(n_components, len(common_channels)) + (len(common_channels) % 2 if n_components > len(common_channels) else 0)
    summary["sample_counts"] = {
        "total": int(len(metadata_df)),
        LEFT: int((metadata_df["label"] == LEFT).sum()),
        RIGHT: int((metadata_df["label"] == RIGHT).sum()),
    }
    return eval_outputs | {"summary": summary}


def _summary_row(label: str, summary: dict[str, Any]) -> dict[str, Any]:
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
        td_row["feature_count"] = "32"
    
    window_sec = 1.5
    
    results_csp_2 = run_classifier(
        output_dir=OUTPUT_ROOT / "csp_n2_1.5s",
        include_files=combo_lr_lrj_files,
        window_sec=window_sec,
        n_components=2,
    )
    row_csp_2 = _summary_row("csp_components=2 (1.5s window)", results_csp_2["summary"])

    results_csp_4 = run_classifier(
        output_dir=OUTPUT_ROOT / "csp_n4_1.5s",
        include_files=combo_lr_lrj_files,
        window_sec=window_sec,
        n_components=4,
    )
    row_csp_4 = _summary_row("csp_components=4 (1.5s window)", results_csp_4["summary"])

    # Actually if we only have 4 channels total, n_components=6 will just crop to 4. We will run it though to test robustness.
    results_csp_6 = run_classifier(
        output_dir=OUTPUT_ROOT / "csp_n6_1.5s",
        include_files=combo_lr_lrj_files,
        window_sec=window_sec,
        n_components=6,
    )
    row_csp_6 = _summary_row("csp_components=6 (max) (1.5s window)", results_csp_6["summary"])

    rows = []
    if td_row:
        rows.append(td_row)
    rows.extend([row_csp_2, row_csp_4, row_csp_6])
    
    scorecard_df = pd.DataFrame(rows)
    for col in ("pooled_accuracy", "pooled_macro_f1"):
        scorecard_df[col] = scorecard_df[col].map(lambda value: f"{float(value):.3f}")

    print("CSP Motor-Imagery Baseline Review\n")
    print(dataframe_to_markdown(scorecard_df))
    
    lines = [
        "# CSP Motor-Imagery Baseline Review",
        "",
        "## Feature Dimension Guardrails",
        f"- Common Channels Active: `{', '.join(results_csp_4['summary']['selected_channels'])}`",
        f"- CSP Extracted Features: Dynamic based on `n_components` clamp.",
        "",
        "## CSP vs Time-Domain Results",
        dataframe_to_markdown(scorecard_df),
        "",
        "### Interpretation",
        "If CSP outperforms the time domain baseline, dimensional spatial filters are definitively required to untangle intent variance.",
    ]
    write_markdown(REPORT_PATH, "\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
