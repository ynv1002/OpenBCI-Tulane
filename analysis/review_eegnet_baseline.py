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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

from analysis.lr_event_classifier.cross_session_evaluator import evaluate_tensor_models
from analysis.lr_event_classifier.event_matrix_builder import (
    LABELS,
    LEFT,
    RIGHT,
    build_tensor_matrix,
)
from analysis.lr_event_classifier.csp_transformer import BandpassFilter
from analysis.lr_event_classifier.eegnet import EEGNet
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
REPORT_PATH = OUTPUT_ROOT / "eegnet_results.md"

class NumpyFloat32Caster(BaseEstimator, TransformerMixin):
    def fit(self, X: np.ndarray, y: Any = None) -> "NumpyFloat32Caster":
        return self
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)

class MappedNeuralNetClassifier(NeuralNetClassifier):
    def fit(self, X, y, **fit_params):
        from sklearn.preprocessing import LabelEncoder
        if not hasattr(self, "le_"):
            self.le_ = LabelEncoder()
        y_int = self.le_.fit_transform(y).astype(np.int64)
        return super().fit(X, y_int, **fit_params)

    def predict(self, X):
        y_pred_int = super().predict(X)
        return self.le_.inverse_transform(y_pred_int)

def _model_bank(n_channels: int, n_times: int) -> dict[str, Any]:
    
    # Early stopping prevents over-training on sparse channel sets across folds.
    eegnet_clf = MappedNeuralNetClassifier(
        module=EEGNet,
        module__n_channels=n_channels,
        module__n_times=n_times,
        module__n_classes=2,
        module__dropout_rate=0.5,
        criterion=nn.CrossEntropyLoss,
        optimizer=optim.Adam,
        optimizer__weight_decay=1e-4,
        lr=1e-3,
        max_epochs=80,
        batch_size=16,
        iterator_train__shuffle=True,
        callbacks=[EarlyStopping(monitor="train_loss", patience=25)],
        train_split=None, # Disable internal Skorch validation split, we are doing outer LOSO fold!
        device="cuda" if torch.cuda.is_available() else "cpu",
        verbose=0
    )

    return {
        "EEGNet-4,2": Pipeline([
            ("bandpass", BandpassFilter(lowcut=8.0, highcut=30.0, fs=250.0)),
            ("float32", NumpyFloat32Caster()),
            ("model", eegnet_clf)
        ])
    }

def run_classifier(
    output_dir: Path,
    include_files: list[str],
    window_sec: float,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)

    print("Building Tensor Matrix...")
    X_tensor, metadata_df, common_channels = build_tensor_matrix(
        include_files=include_files,
        window_sec=window_sec,
    )
    
    n_trials, n_channels, n_times = X_tensor.shape
    print(f"Executing Skorch Pipeline on tensor shape {X_tensor.shape} across {len(common_channels)} channels...")

    # Skorch requires explicit int64 targets for CrossEntropyLoss mapping natively,
    # but strictly speaking `cross_session_evaluator.py` expects predictions as string `labels`! 
    # Luckily, `NeuralNetClassifier` natively intercepts string `y_train` via an internal `LabelEncoder`
    # and properly maps the `predict()` output back to strings automatically.

    eval_outputs = evaluate_tensor_models(
        X_tensor=X_tensor,
        metadata_df=metadata_df,
        model_bank=_model_bank(n_channels=n_channels, n_times=n_times),
        labels=LABELS,
        output_dir=output_dir,
    )

    summary = eval_outputs["summary"]
    summary["files_used"] = sorted(metadata_df["file"].unique().tolist())
    summary["selected_channels"] = common_channels
    summary["feature_count"] = "Deep Network Representation"
    summary["sample_counts"] = {
        "total": int(len(metadata_df)),
        LEFT: int((metadata_df["label"] == LEFT).sum()),
        RIGHT: int((metadata_df["label"] == RIGHT).sum()),
    }
    return eval_outputs | {"summary": summary}


def _summary_row(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_set": label,
        "feature_count": str(summary.get("feature_count", "N/A")),
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
    
    results_eegnet = run_classifier(
        output_dir=OUTPUT_ROOT / "eegnet_1.5s",
        include_files=combo_lr_lrj_files,
        window_sec=window_sec,
    )
    row_eegnet = _summary_row("EEGNet-4,2 (1.5s window)", results_eegnet["summary"])

    rows = []
    if td_row:
        rows.append(td_row)
        
    rows.append({
        "evaluation_set": "csp_components=2 (Phase 3 Baseline)",
        "feature_count": "2",
        "event_count": 771,
        "best_model": "LogisticRegression",
        "pooled_accuracy": 0.546,
        "pooled_macro_f1": 0.546
    })
    
    rows.append(row_eegnet)
    
    scorecard_df = pd.DataFrame(rows)
    for col in ("pooled_accuracy", "pooled_macro_f1"):
        if col in scorecard_df.columns:
            scorecard_df[col] = scorecard_df[col].map(lambda value: f"{float(value):.3f}")

    print("EEGNet Deep Learning Motor-Imagery Review\n")
    print(dataframe_to_markdown(scorecard_df))
    
    lines = [
        "# EEGNet Deep Learning Motor-Imagery Review",
        "",
        "## Deep Representational Shift",
        f"- Common Channels Active: `{', '.join(results_eegnet['summary']['selected_channels'])}`",
        f"- Target Classes: `Left vs Right`",
        "",
        "## EEGNet vs CSP vs Legacy Time-Domain",
        dataframe_to_markdown(scorecard_df),
        "",
        "### Interpretation",
        "If EEGNet bridges the gap from 0.546 (CSP) toward or surpassing 0.628, the inclusion of temporal convolutions actively unmasks intentionality invisible to simple geometric bounds.",
    ]
    write_markdown(REPORT_PATH, "\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")

if __name__ == "__main__":
    main()
