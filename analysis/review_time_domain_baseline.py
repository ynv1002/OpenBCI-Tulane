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

from analysis.lr_event_classifier.cross_session_evaluator import evaluate_models
from analysis.lr_event_classifier.event_matrix_builder import (
    LABELS,
    LEFT,
    RIGHT,
    build_event_matrix,
)
from analysis.stepwise_protocol_registry import (
    all_file_contracts,
    eeg_lr_main_contracts,
    eeg_lr_stress_contracts,
    lrj_main_contracts,
)
from analysis.utils import (
    PROJECT_ROOT,
    dataframe_to_markdown,
    ensure_output_dir,
    write_json,
    write_markdown,
)

OUTPUT_ROOT = PROJECT_ROOT / "analysis" / "outputs"
SUMMARY_PATH = OUTPUT_ROOT / "left_right_coverage_summary.json"
INVENTORY_PATH = OUTPUT_ROOT / "left_right_relevant_files.csv"
REPORT_PATH = OUTPUT_ROOT / "left_right_lr_results.md"

VALIDATION_SUMMARY_PATH = PROJECT_ROOT / "analysis" / "lr_event_validation" / "outputs" / "overall_summary.csv"
LRJ_BENCHMARK_RESULTS_PATH = PROJECT_ROOT / "analysis" / "lrj_benchmark" / "outputs" / "benchmark_results.csv"
HAND_LRJ_SUMMARY_PATH = PROJECT_ROOT / "analysis" / "hand_movement_lrj" / "outputs" / "summary.json"


def _per_channel_features(values: np.ndarray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(values**2))),
        "mav": float(np.mean(np.abs(values))),
        "variance": float(np.var(values)),
        "peak_abs": float(np.max(np.abs(values))),
        "waveform_length": float(np.sum(np.abs(np.diff(values)))) if len(values) > 1 else 0.0,
    }


def _aggregate_rms(window: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(window**2, axis=1))


def _time_domain_event_features(window: np.ndarray, channel_columns: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    channel_rms: dict[str, float] = {}
    channel_mav: dict[str, float] = {}
    for idx, channel in enumerate(channel_columns):
        channel_values = window[:, idx]
        channel_features = _per_channel_features(channel_values)
        channel_rms[channel] = channel_features["rms"]
        channel_mav[channel] = channel_features["mav"]
        for suffix, value in channel_features.items():
            features[f"{channel}_{suffix}"] = value

    mean_rms = float(np.mean(list(channel_rms.values())))
    mean_mav = float(np.mean(list(channel_mav.values())))
    for channel in channel_columns:
        features[f"{channel}_rms_asym"] = float(channel_rms[channel] - mean_rms)
        features[f"{channel}_mav_asym"] = float(channel_mav[channel] - mean_mav)

    aggregate = _aggregate_rms(window)
    features["aggregate_rms_mean"] = float(np.mean(aggregate))
    features["aggregate_rms_std"] = float(np.std(aggregate))
    features["aggregate_rms_max"] = float(np.max(aggregate))
    features["aggregate_rms_waveform_length"] = (
        float(np.sum(np.abs(np.diff(aggregate)))) if len(aggregate) > 1 else 0.0
    )
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
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)

    event_matrix_df, feature_columns, common_channels = build_event_matrix(
        feature_extractor=_time_domain_event_features,
        include_files=include_files,
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
    summary["sample_counts"] = {
        "total": int(len(event_matrix_df)),
        LEFT: int((event_matrix_df["label"] == LEFT).sum()),
        RIGHT: int((event_matrix_df["label"] == RIGHT).sum()),
    }
    return eval_outputs | {"summary": summary}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _lr_inventory_dataframe(validation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    
    file_stats = {}
    for row in validation_df.to_dict(orient="records"):
        file_stats[str(row["file"])] = {
            "inside_block_fraction": float(row["inside_block_fraction"]),
            "kept_events_inside_blocks": int(row["kept_events_inside_blocks"]),
            "kept_events_outside_blocks": int(row["kept_events_outside_blocks"]),
        }

    for contract in all_file_contracts():
        if contract.protocol_family in ("LR", "LRJ"):
            stats = file_stats.get(contract.filename, {})
            current_use = "Diagnostic only"
            if contract.trust_tier == "main":
                current_use = (
                    "Primary LR event-model scorecard and expanded coverage"
                    if contract.protocol_family == "LR"
                    else "Separate LRJ benchmark only"
                )
            elif contract.trust_tier == "stress":
                current_use = "Expanded LR event coverage only"

            include_in = "yes" if contract.protocol_family == "LR" else "no"

            rows.append(
                {
                    "filename": contract.filename,
                    "protocol_family": contract.protocol_family,
                    "trust_tier": contract.trust_tier,
                    "current_left_right_use": current_use,
                    "include_in_current_lr_event_model": include_in,
                    "inside_block_fraction": stats.get("inside_block_fraction", ""),
                    "kept_events_inside_blocks": stats.get("kept_events_inside_blocks", ""),
                    "kept_events_outside_blocks": stats.get("kept_events_outside_blocks", ""),
                }
            )
            
    inventory_df = pd.DataFrame(rows)
    inventory_df.to_csv(INVENTORY_PATH, index=False)
    return inventory_df


def _summary_row(label: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_set": label,
        "files_used": ", ".join(summary["files_used"]),
        "selected_channels": ", ".join(summary["selected_channels"]),
        "event_count": int(summary["sample_counts"]["total"]),
        "left_events": int(summary["sample_counts"][LEFT]),
        "right_events": int(summary["sample_counts"][RIGHT]),
        "best_model": str(summary["best_model"]["name"]),
        "pooled_accuracy": float(summary["best_model"]["accuracy"]),
        "pooled_macro_f1": float(summary["best_model"]["macro_f1"]),
    }


def _best_fold_table(results_path: Path, best_model_name: str) -> pd.DataFrame:
    results_df = pd.read_csv(results_path)
    fold_df = results_df[
        (results_df["model_name"] == best_model_name) & (results_df["train_file"] != "__pooled_cross_session__")
    ].copy()
    if fold_df.empty:
        return pd.DataFrame()
    return fold_df.loc[:, ["train_file", "test_file", "accuracy", "macro_f1"]].reset_index(drop=True)


def _lrj_summary_rows() -> tuple[dict[str, Any], dict[str, Any]]:
    if not LRJ_BENCHMARK_RESULTS_PATH.exists():
        return ({}, {})
        
    lrj_results_df = pd.read_csv(LRJ_BENCHMARK_RESULTS_PATH)
    pooled_lrj = lrj_results_df[lrj_results_df["evaluation_scope"] == "pooled"].copy()
    best_movement = pooled_lrj.sort_values(
        ["movement_interval_macro_f1", "movement_window_macro_f1"],
        ascending=False,
    ).iloc[0]

    hand_summary = _load_json(HAND_LRJ_SUMMARY_PATH)
    if not hand_summary:
        return ({}, {})
        
    return (
        {
            "benchmark": "Shared LRJ movement benchmark",
            "files_used": str(best_movement["train_files"]),
            "best_model": str(best_movement["model_name"]),
            "primary_metric": float(best_movement["movement_interval_macro_f1"]),
            "secondary_metric": float(best_movement["movement_window_macro_f1"]),
        },
        {
            "benchmark": "Hand LRJ exact-count branch",
            "files_used": ", ".join(row["filename"] for row in hand_summary["per_file_rows"]),
            "best_model": "existing hand-branch decode",
            "primary_metric": float(hand_summary["pooled_accuracy"]),
            "secondary_metric": float(hand_summary["pooled_mae"]),
        },
    )


def main() -> None:
    ensure_output_dir(OUTPUT_ROOT)
    main_lr_files = [contract.filename for contract in eeg_lr_main_contracts()]
    all_lr_files = main_lr_files + [contract.filename for contract in eeg_lr_stress_contracts()]
    combo_lr_lrj_files = all_lr_files + [contract.filename for contract in lrj_main_contracts()]

    validation_df = pd.read_csv(VALIDATION_SUMMARY_PATH)
    inventory_df = _lr_inventory_dataframe(validation_df)

    main_results = run_classifier(
        output_dir=OUTPUT_ROOT / "strict_main_lr_coverage",
        include_files=main_lr_files,
    )
    main_summary = main_results["summary"]

    expanded_results = run_classifier(
        output_dir=OUTPUT_ROOT / "expanded_all_lr_coverage",
        include_files=all_lr_files,
    )
    expanded_summary = expanded_results["summary"]

    combo_results = run_classifier(
        output_dir=OUTPUT_ROOT / "combo_lr_lrj_coverage",
        include_files=combo_lr_lrj_files,
    )
    combo_summary = combo_results["summary"]

    scorecard_df = pd.DataFrame(
        [
            _summary_row("strict_main_lr", main_summary),
            _summary_row("expanded_all_lr", expanded_summary),
            _summary_row("combo_lr_lrj", combo_summary),
        ]
    )
    scorecard_preview = scorecard_df.copy()
    for column in ("pooled_accuracy", "pooled_macro_f1"):
        scorecard_preview[column] = scorecard_preview[column].map(lambda value: f"{float(value):.3f}")

    validation_preview = validation_df[validation_df["file"].isin(all_lr_files)].loc[
        :,
        [
            "file",
            "inside_block_fraction",
            "kept_events_inside_blocks",
            "kept_events_outside_blocks",
            "count_channels",
        ],
    ].copy()
    validation_preview["inside_block_fraction"] = validation_preview["inside_block_fraction"].map(
        lambda value: f"{float(value):.3f}"
    )

    expanded_fold_preview = _best_fold_table(
        OUTPUT_ROOT / "expanded_all_lr_coverage" / "classifier_results.csv",
        str(expanded_summary["best_model"]["name"]),
    )
    if not expanded_fold_preview.empty:
        for column in ("accuracy", "macro_f1"):
            expanded_fold_preview[column] = expanded_fold_preview[column].map(lambda value: f"{float(value):.3f}")

    inventory_preview = inventory_df.loc[
        :,
        [
            "filename",
            "protocol_family",
            "trust_tier",
            "current_left_right_use",
            "include_in_current_lr_event_model",
        ],
    ].copy()

    lrj_rows = list(_lrj_summary_rows())
    lrj_preview = pd.DataFrame(lrj_rows) if lrj_rows[0] else pd.DataFrame()
    if not lrj_preview.empty:
        lrj_preview["primary_metric"] = lrj_preview["primary_metric"].map(lambda value: f"{float(value):.3f}")
        lrj_preview["secondary_metric"] = lrj_preview["secondary_metric"].map(lambda value: f"{float(value):.3f}")

    summary_payload = {
        "main_lr_files": main_lr_files,
        "all_lr_files": all_lr_files,
        "main_lr_summary": main_summary,
        "expanded_lr_summary": expanded_summary,
        "combo_lr_lrj_summary": combo_summary,
        "lrj_related_summaries": lrj_rows,
        "inventory_csv": str(INVENTORY_PATH),
    }
    write_json(SUMMARY_PATH, summary_payload)

    lines = [
        "# Left/Right Time-Domain Baseline Review",
        "",
        "## Recommendation",
        f"- The current LR event classifier should expand from the original two-file scorecard to all four Yaniv `LR` runs: `{', '.join(all_lr_files)}`.",
        "- We have also executed a `combo_lr_lrj` pass folding in the `LRJ` benchmark files to test whether training on repeating pulse-blocks improves cross-session transfer.",
        "- The Dalin `LR` / `LRJ` files should remain diagnostic-only for now.",
        "",
        "## Relevant Files",
        dataframe_to_markdown(inventory_preview),
        "",
        "## LR Event-Level Coverage",
        dataframe_to_markdown(scorecard_preview),
        "",
        "### Per-File LR Validation",
        dataframe_to_markdown(validation_preview),
        "",
        f"- Expanded all-LR run selected channels: `{', '.join(expanded_summary['selected_channels'])}`",
        f"- Expanded all-LR best pooled model: `{expanded_summary['best_model']['name']}` with accuracy `{expanded_summary['best_model']['accuracy']:.3f}` and macro-F1 `{expanded_summary['best_model']['macro_f1']:.3f}`.",
        "",
        "### Best Expanded LR Fold Results",
        dataframe_to_markdown(expanded_fold_preview) if not expanded_fold_preview.empty else "No fold results available.",
        "",
        "## Separate LRJ Evidence",
        dataframe_to_markdown(lrj_preview) if not lrj_preview.empty else "No LRJ evidence available.",
        "",
        "## Scope Decision",
        "- Include now in the LR event-model coverage: all four Yaniv `LR` files.",
        "- Keep as separate supporting left/right evidence: `Ben-LRJ(1-6)-4:9.csv`, `Yaniv-LRJ(6)-4:7.csv`.",
        "- Exclude from the report scorecard: `Dalin-LR(6)-4:7.csv`, `Dalin-LRJ(1,2)-4:7.csv`.",
    ]
    write_markdown(REPORT_PATH, "\n".join(lines))

    print("Time-Domain Left/Right Baseline run successfully.")
    print(f"  Report: {REPORT_PATH}")
    print("")
    print("Baseline scorecard")
    print(dataframe_to_markdown(scorecard_preview))


if __name__ == "__main__":
    main()
