from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .event_utils import (
    EVENT_ACTIVE,
    EVENT_INACTIVE,
    EVENT_OFFSET,
    EVENT_ONSET,
    EventWindowConfig,
    JawEventConfig,
    _linear_slope,
)
from .live_jaw_click_detector import (
    ReplayEvaluationConfig,
    build_default_trigger_grid,
    build_jaw_replay_dataset,
    build_replay_score_frame,
    choose_best_config,
    evaluate_clicks_against_reference,
    export_strategy_artifacts,
    extract_reference_onsets,
    run_trigger_strategy,
    summarize_strategy_performance,
    train_jaw_click_models,
    write_best_strategy_report,
)
from .lr_event_classifier.review_lr_event_classifier import EVENT_WINDOW_SEC, _event_features, _window_bounds
from .lr_event_validation.review_yaniv_lr import (
    FROZEN_LR6,
    LEFT,
    RIGHT,
    _build_block_table,
    _build_file_outputs,
    _summarize_file,
    _target_count_channels,
    _write_plot,
)
from .lrj_dataset import DEFAULT_LRJ_FS, LRJSessionSpec, build_lrj_session_record
from .stepwise_protocol_registry import (
    FileContract,
    all_file_contracts,
    diagnostic_contracts,
    eeg_lr_main_contracts,
    eeg_lr_stress_contracts,
    jaw_hr_contracts,
    lrj_main_contracts,
    registry_dataframe,
)
from .utils import (
    audit_session,
    collapse_marker_events,
    compute_channel_quality,
    dataframe_to_markdown,
    ensure_output_dir,
    estimate_sampling,
    load_openbci_csv,
    preprocess_session_signals,
    write_json,
    write_markdown,
)


JAW_TYPE_LABELS = ["HOLD", "REPEATED"]
EEG_SIDE_LABELS = [LEFT, RIGHT]
JAW_STAGE1_GATE = {"event_f1": 0.72, "precision": 0.68, "recall": 0.68}
JAW_STAGE2_GATE = {"macro_f1": 0.75}
JAW_STAGE3_GATE = {"exact_accuracy": 0.60, "per_file_floor": 0.45, "mae": 1.0}
EEG_STAGE12_GATE = {
    "per_file_inside_fraction": 0.94,
    "overall_inside_fraction": 0.90,
    "lrj_exact_accuracy": 0.55,
    "lrj_mae": 1.0,
}
EEG_STAGE3_GATE = {"macro_f1": 0.58, "class_recall": 0.50}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the stage-gated jaw + EEG benchmark using explicit per-file protocol contracts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "stepwise_benchmark",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=DEFAULT_LRJ_FS,
        help="Authoritative reporting sample rate.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip EEG validation plots.",
    )
    return parser.parse_args()


def _channel_index(channel_name: str) -> int:
    return int(str(channel_name).split("_")[-1])


def _file_sort_key(contract: FileContract) -> tuple[str | None, int | None, str]:
    audit = audit_session(contract.csv_path.resolve(), "jaw" if contract.processing_family == "jaw" else "left_right")
    parsed_date = audit.get("parsed_date")
    run_index = audit.get("run_index")
    return (str(parsed_date) if parsed_date else "", int(run_index) if run_index is not None else 999, contract.filename)


def _validate_registry(output_dir: Path) -> pd.DataFrame:
    contracts = all_file_contracts()
    keys = [contract.key for contract in contracts]
    if len(keys) != len(set(keys)):
        raise ValueError("Manual protocol registry contains duplicate keys.")
    for contract in contracts:
        if not contract.csv_path.exists():
            raise ValueError(f"Registry path does not exist: {contract.csv_path}")
    registry_df = registry_dataframe(contracts)
    registry_df.to_csv(output_dir / "protocol_registry.csv", index=False)
    write_json(
        output_dir / "protocol_registry.json",
        {"contracts": registry_df.to_dict(orient="records")},
    )
    return registry_df


def _ordered_manual_audits(contracts: Sequence[FileContract], family_key: str) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for contract in contracts:
        audit = audit_session(contract.csv_path.resolve(), family_key)
        audit["contract_key"] = contract.key
        audit["trust_tier"] = contract.trust_tier
        audit["protocol_family"] = contract.protocol_family
        audit["marker_semantics"] = contract.marker_semantics
        audits.append(audit)
    audits.sort(key=lambda audit: (str(audit.get("parsed_date") or ""), int(audit.get("run_index") or 999), audit["filename"]))
    for session_rank, audit in enumerate(audits, start=1):
        audit["session_rank"] = session_rank
    return audits


def _simple_classifier_bank() -> dict[str, Any]:
    return {
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]
        ),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
    }


def _format_float(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{float(value):.3f}"


def _render_stage_header(title: str, passed: bool) -> str:
    return f"# {title}\n\n- Gate pass: `{passed}`\n"


def _write_stage_summary(output_dir: Path, summary_name: str, lines: list[str], payload: dict[str, Any]) -> None:
    write_markdown(output_dir / f"{summary_name}.md", "\n".join(lines))
    write_json(output_dir / f"{summary_name}.json", payload)


def _write_skipped_stage(output_dir: Path, title: str, reason: str) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    payload = {"gate_pass": False, "skipped": True, "reason": reason}
    lines = [
        f"# {title}",
        "",
        "- Gate pass: `False`",
        f"- Skipped: `{reason}`",
    ]
    _write_stage_summary(output_dir, "summary", lines, payload)
    return payload


def _run_jaw_stage1(
    contracts: Sequence[FileContract],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    audits = _ordered_manual_audits(contracts, "jaw")
    event_config = JawEventConfig()
    window_config = EventWindowConfig()
    evaluation_config = ReplayEvaluationConfig()

    dataset_bundle = build_jaw_replay_dataset(audits, event_config, window_config)
    model_bundle = train_jaw_click_models(dataset_bundle)
    score_frame = build_replay_score_frame(dataset_bundle, model_bundle)
    reference_onsets = extract_reference_onsets(dataset_bundle["event_bundle"]["interval_frame"])

    dataset_bundle["event_bundle"]["sample_frame"].to_csv(output_dir / "jaw_event_samples.csv", index=False)
    dataset_bundle["event_bundle"]["interval_frame"].to_csv(output_dir / "jaw_event_intervals.csv", index=False)
    dataset_bundle["event_bundle"]["coarse_segment_frame"].to_csv(
        output_dir / "jaw_coarse_segments.csv",
        index=False,
    )
    dataset_bundle["event_bundle"]["threshold_frame"].to_csv(output_dir / "jaw_event_thresholds.csv", index=False)
    score_frame.to_csv(output_dir / "jaw_replay_scores.csv", index=False)
    reference_onsets.to_csv(output_dir / "jaw_reference_onsets.csv", index=False)

    tuning_rows: list[dict[str, Any]] = []
    for trigger_config in build_default_trigger_grid():
        click_frame = run_trigger_strategy(score_frame, trigger_config)
        summary_df, _ = evaluate_clicks_against_reference(click_frame, reference_onsets, evaluation_config)
        tuning_rows.append(
            summarize_strategy_performance(summary_df, trigger_config.strategy_name, trigger_config)
        )
    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["strategy_name", "train_weighted_event_f1", "train_weighted_precision", "test_weighted_event_f1"],
        ascending=[True, False, False, False],
    )
    tuning_df.to_csv(output_dir / "jaw_trigger_tuning.csv", index=False)

    final_rows: list[dict[str, Any]] = []
    for strategy_name in tuning_df["strategy_name"].drop_duplicates():
        strategy_candidates = tuning_df[tuning_df["strategy_name"] == strategy_name].to_dict(orient="records")
        best_strategy_summary = choose_best_config(strategy_candidates)
        trigger_config = best_strategy_summary["trigger_config"]
        click_frame = run_trigger_strategy(score_frame, type(build_default_trigger_grid()[0])(**trigger_config))
        summary_df, match_df = evaluate_clicks_against_reference(click_frame, reference_onsets, evaluation_config)
        strategy_summary = summarize_strategy_performance(
            summary_df,
            strategy_name,
            type(build_default_trigger_grid()[0])(**trigger_config),
        )
        strategy_summary.update(
            export_strategy_artifacts(
                output_dir,
                strategy_name,
                score_frame,
                click_frame,
                summary_df,
                match_df,
            )
        )
        final_rows.append(strategy_summary)

    final_df = pd.DataFrame(final_rows).sort_values(
        ["test_weighted_event_f1", "test_weighted_precision", "train_weighted_event_f1"],
        ascending=[False, False, False],
    )
    final_df.to_csv(output_dir / "jaw_stage1_results.csv", index=False)
    best_row = final_df.iloc[0].to_dict()
    write_best_strategy_report(
        output_dir / "jaw_stage1_best_strategy.md",
        output_dir / "jaw_stage1_best_strategy.json",
        best_row,
        final_df,
        evaluation_config,
    )

    gate_pass = (
        float(best_row["test_weighted_event_f1"]) >= JAW_STAGE1_GATE["event_f1"]
        and float(best_row["test_weighted_precision"]) >= JAW_STAGE1_GATE["precision"]
        and float(best_row["test_weighted_recall"]) >= JAW_STAGE1_GATE["recall"]
    )
    payload = {
        "gate_pass": gate_pass,
        "gate_thresholds": JAW_STAGE1_GATE,
        "event_config": asdict(event_config),
        "window_config": asdict(window_config),
        "evaluation_config": asdict(evaluation_config),
        "selected_channels": model_bundle["selected_channels"],
        "excluded_channels": model_bundle["excluded_channels"],
        "best_strategy": best_row,
        "strategy_rows": final_df.to_dict(orient="records"),
    }
    lines = [
        _render_stage_header("Jaw Stage 1: Event Detection", gate_pass),
        f"- Train files: `{', '.join(audit['filename'] for audit in audits[:2])}`",
        f"- Test file: `{audits[-1]['filename']}`",
        f"- Selected channels: `{', '.join(model_bundle['selected_channels'])}`",
        f"- Best strategy: `{best_row['strategy_name']}`",
        f"- Test weighted event-F1: `{_format_float(best_row['test_weighted_event_f1'])}`",
        f"- Test weighted precision: `{_format_float(best_row['test_weighted_precision'])}`",
        f"- Test weighted recall: `{_format_float(best_row['test_weighted_recall'])}`",
        "",
        "## Strategy Summary",
        "",
        dataframe_to_markdown(
            final_df[
                [
                    "strategy_name",
                    "test_weighted_event_f1",
                    "test_weighted_precision",
                    "test_weighted_recall",
                    "test_total_references",
                    "test_total_clicks",
                    "test_total_extra_clicks",
                ]
            ],
            include_index=False,
        ),
    ]
    _write_stage_summary(output_dir, "summary", lines, payload)
    payload["dataset_bundle"] = dataset_bundle
    return payload


def _build_jaw_type_feature_table(event_bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    for session_bundle in event_bundle["session_bundles"]:
        sample_frame = session_bundle["sample_frame"].reset_index(drop=True)
        segment_frame = session_bundle["coarse_segment_frame"].copy()
        active_segments = segment_frame[segment_frame["coarse_label"].isin(JAW_TYPE_LABELS)]
        for segment in active_segments.to_dict(orient="records"):
            start_sample = int(segment["start_sample"])
            end_sample = int(segment["end_sample"])
            segment_samples = sample_frame.iloc[start_sample : end_sample + 1]
            aggregate = segment_samples["aggregate_rms"].to_numpy(dtype=float)
            aggregate_smooth = segment_samples["aggregate_rms_smooth"].to_numpy(dtype=float)
            event_labels = segment_samples["event_label"].astype(str)
            row = {
                "filename": str(segment["filename"]),
                "file_path": str(segment["file_path"]),
                "parsed_date": str(segment["parsed_date"]),
                "session_rank": int(segment["session_rank"]),
                "split_role": str(segment["split_role"]),
                "label": str(segment["coarse_label"]),
                "coarse_segment_index": int(segment["coarse_segment_index"]),
                "duration_sec": float(segment["duration_sec"]),
                "detected_peak_count": int(segment["detected_peak_count"]),
                "detected_event_count": int(segment["detected_event_count"]),
                "fallback_used": bool(segment["fallback_used"]),
                "active_threshold": float(segment["active_threshold"]),
                "release_threshold": float(segment["release_threshold"]),
                "peak_prominence_threshold": float(segment["peak_prominence_threshold"]),
                "onset_fraction": float((event_labels == EVENT_ONSET).mean()),
                "active_fraction": float((event_labels == EVENT_ACTIVE).mean()),
                "offset_fraction": float((event_labels == EVENT_OFFSET).mean()),
                "inactive_fraction": float((event_labels == EVENT_INACTIVE).mean()),
            }
            for prefix, values in (("aggregate_rms", aggregate), ("aggregate_rms_smooth", aggregate_smooth)):
                row[f"{prefix}_mean"] = float(np.mean(values))
                row[f"{prefix}_std"] = float(np.std(values))
                row[f"{prefix}_max"] = float(np.max(values))
                row[f"{prefix}_delta"] = float(values[-1] - values[0])
                row[f"{prefix}_ptp"] = float(np.ptp(values))
                row[f"{prefix}_slope"] = _linear_slope(values)
            rows.append(row)
    feature_df = pd.DataFrame(rows).sort_values(["filename", "coarse_segment_index"]).reset_index(drop=True)
    metadata_columns = {
        "filename",
        "file_path",
        "parsed_date",
        "session_rank",
        "split_role",
        "label",
        "coarse_segment_index",
    }
    feature_columns = [column for column in feature_df.columns if column not in metadata_columns]
    return feature_df, feature_columns


def _run_jaw_type_stage(
    event_bundle: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    feature_df, feature_columns = _build_jaw_type_feature_table(event_bundle)
    feature_df.to_csv(output_dir / "jaw_type_segment_features.csv", index=False)
    train_df = feature_df[feature_df["split_role"] == "train"].reset_index(drop=True)
    test_df = feature_df[feature_df["split_role"] == "test"].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    heuristic_pred = np.where(test_df["detected_event_count"].to_numpy(dtype=int) > 1, "REPEATED", "HOLD")
    heuristic_cm = confusion_matrix(test_df["label"], heuristic_pred, labels=JAW_TYPE_LABELS)
    heuristic_cm_path = output_dir / "confusion_jaw_type_heuristic.csv"
    pd.DataFrame(
        heuristic_cm,
        index=[f"true_{label}" for label in JAW_TYPE_LABELS],
        columns=[f"pred_{label}" for label in JAW_TYPE_LABELS],
    ).to_csv(heuristic_cm_path)
    rows.append(
        {
            "model_name": "CountHeuristic",
            "accuracy": float(accuracy_score(test_df["label"], heuristic_pred)),
            "macro_f1": float(f1_score(test_df["label"], heuristic_pred, labels=JAW_TYPE_LABELS, average="macro", zero_division=0)),
            "confusion_matrix_csv": str(heuristic_cm_path.resolve()),
        }
    )
    prediction_frames.append(
        test_df.loc[:, ["filename", "coarse_segment_index", "label"]]
        .rename(columns={"label": "true_label"})
        .assign(model_name="CountHeuristic", pred_label=heuristic_pred)
    )

    X_train = train_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy()
    for model_name, model in _simple_classifier_bank().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        confusion_path = output_dir / f"confusion_jaw_type_{model_name}.csv"
        pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=JAW_TYPE_LABELS),
            index=[f"true_{label}" for label in JAW_TYPE_LABELS],
            columns=[f"pred_{label}" for label in JAW_TYPE_LABELS],
        ).to_csv(confusion_path)
        rows.append(
            {
                "model_name": model_name,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, labels=JAW_TYPE_LABELS, average="macro", zero_division=0)),
                "confusion_matrix_csv": str(confusion_path.resolve()),
            }
        )
        prediction_frames.append(
            test_df.loc[:, ["filename", "coarse_segment_index", "label"]]
            .rename(columns={"label": "true_label"})
            .assign(model_name=model_name, pred_label=y_pred)
        )

    results_df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    results_df.to_csv(output_dir / "jaw_type_results.csv", index=False)
    predictions_df.to_csv(output_dir / "jaw_type_predictions.csv", index=False)

    best_row = results_df.iloc[0].to_dict()
    gate_pass = float(best_row["macro_f1"]) >= JAW_STAGE2_GATE["macro_f1"]
    payload = {
        "gate_pass": gate_pass,
        "gate_thresholds": JAW_STAGE2_GATE,
        "best_model": best_row,
        "result_rows": results_df.to_dict(orient="records"),
    }
    lines = [
        _render_stage_header("Jaw Stage 2: HOLD vs REPEATED", gate_pass),
        f"- Train files: `{', '.join(sorted(train_df['filename'].unique()))}`",
        f"- Test file: `{', '.join(sorted(test_df['filename'].unique()))}`",
        f"- Best model: `{best_row['model_name']}`",
        f"- Best macro-F1: `{_format_float(best_row['macro_f1'])}`",
        "",
        "## Model Summary",
        "",
        dataframe_to_markdown(results_df, include_index=False),
    ]
    _write_stage_summary(output_dir, "summary", lines, payload)
    return payload


def _select_constrained_peaks(candidate_df: pd.DataFrame, expected_count: int) -> pd.DataFrame:
    if candidate_df.empty:
        return candidate_df.copy()
    ranked = candidate_df.sort_values(["peak_prominence", "peak_value"], ascending=[False, False])
    selected = ranked.head(expected_count).sort_values("peak_time_sec").reset_index(drop=True)
    return selected


def _build_lrj_decode_rows(session_records: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in session_records:
        trial_summary_df = record["trial_summary_df"].copy()
        event_candidates_df = record["event_candidates_df"].copy()
        for trial in trial_summary_df.to_dict(orient="records"):
            trial_candidates = event_candidates_df[
                (event_candidates_df["trial_index_overall"] == int(trial["trial_index_overall"]))
                & (event_candidates_df["label"] == str(trial["label"]))
                & (event_candidates_df["kept_for_count"] == True)
            ].copy()
            selected_candidates = _select_constrained_peaks(trial_candidates, int(trial["expected_count"]))
            unconstrained_count = int(trial["observed_count"])
            constrained_count = int(len(selected_candidates))
            expected_count = int(trial["expected_count"])
            rows.append(
                {
                    "subject": record["subject"],
                    "display_name": record["display_name"],
                    "filename": record["filename"],
                    "label": str(trial["label"]),
                    "trial_index_overall": int(trial["trial_index_overall"]),
                    "trial_index_within_label": int(trial["trial_index_within_label"]),
                    "expected_count": expected_count,
                    "candidate_pool_count": int(len(trial_candidates)),
                    "unconstrained_count": unconstrained_count,
                    "unconstrained_error": int(unconstrained_count - expected_count),
                    "unconstrained_exact": bool(unconstrained_count == expected_count),
                    "constrained_count": constrained_count,
                    "constrained_error": int(constrained_count - expected_count),
                    "constrained_exact": bool(constrained_count == expected_count),
                    "start_time_sec": float(trial["start_time_sec"]),
                    "end_time_sec": float(trial["end_time_sec"]),
                    "duration_sec": float(trial["duration_sec"]),
                    "unconstrained_peak_times_sec": json.dumps(
                        [round(float(value), 3) for value in trial_candidates["peak_time_sec"].tolist()]
                    ),
                    "constrained_peak_times_sec": json.dumps(
                        [round(float(value), 3) for value in selected_candidates["peak_time_sec"].tolist()]
                    ),
                    "count_basis": str(trial["count_basis"]),
                    "issues": str(trial["issues"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["filename", "trial_index_overall"]).reset_index(drop=True)


def _summarize_lrj_decode(decode_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (filename, label), group_df in decode_df.groupby(["filename", "label"], sort=False):
        rows.append(
            {
                "filename": filename,
                "label": label,
                "trial_count": int(len(group_df)),
                "unconstrained_exact_accuracy": float(group_df["unconstrained_exact"].mean()),
                "constrained_exact_accuracy": float(group_df["constrained_exact"].mean()),
                "unconstrained_mae": float(np.mean(np.abs(group_df["unconstrained_error"].to_numpy(dtype=float)))),
                "constrained_mae": float(np.mean(np.abs(group_df["constrained_error"].to_numpy(dtype=float)))),
            }
        )
    for label, group_df in decode_df.groupby("label", sort=False):
        rows.append(
            {
                "filename": "__pooled__",
                "label": label,
                "trial_count": int(len(group_df)),
                "unconstrained_exact_accuracy": float(group_df["unconstrained_exact"].mean()),
                "constrained_exact_accuracy": float(group_df["constrained_exact"].mean()),
                "unconstrained_mae": float(np.mean(np.abs(group_df["unconstrained_error"].to_numpy(dtype=float)))),
                "constrained_mae": float(np.mean(np.abs(group_df["constrained_error"].to_numpy(dtype=float)))),
            }
        )
    return pd.DataFrame(rows)


def _build_lrj_decode_bundle(
    contracts: Sequence[FileContract],
    fs_hz: float,
) -> dict[str, Any]:
    session_records = []
    for contract in contracts:
        session_records.append(
            build_lrj_session_record(
                LRJSessionSpec(
                    subject=contract.subject,
                    display_name=f"{contract.subject} LRJ",
                    filename=contract.filename,
                    csv_path=contract.csv_path.resolve(),
                ),
                fs_hz=fs_hz,
            )
        )
    decode_df = _build_lrj_decode_rows(session_records)
    summary_df = _summarize_lrj_decode(decode_df)
    return {
        "decode_df": decode_df,
        "summary_df": summary_df,
        "contracts": [contract.filename for contract in contracts],
    }


def _run_jaw_lrj_stage(
    lrj_bundle: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    decode_df = lrj_bundle["decode_df"]
    summary_df = lrj_bundle["summary_df"]
    jaw_decode_df = decode_df[decode_df["label"] == "JAW"].copy().reset_index(drop=True)
    jaw_summary_df = summary_df[summary_df["label"] == "JAW"].copy().reset_index(drop=True)
    jaw_decode_df.to_csv(output_dir / "jaw_lrj_trial_decode.csv", index=False)
    jaw_summary_df.to_csv(output_dir / "jaw_lrj_decode_summary.csv", index=False)

    pooled_jaw = jaw_summary_df[jaw_summary_df["filename"] == "__pooled__"].iloc[0]
    per_file_jaw = jaw_summary_df[jaw_summary_df["filename"] != "__pooled__"].copy()
    gate_pass = (
        float(pooled_jaw["constrained_exact_accuracy"]) >= JAW_STAGE3_GATE["exact_accuracy"]
        and float(pooled_jaw["constrained_mae"]) <= JAW_STAGE3_GATE["mae"]
        and bool((per_file_jaw["constrained_exact_accuracy"] >= JAW_STAGE3_GATE["per_file_floor"]).all())
    )
    payload = {
        "gate_pass": gate_pass,
        "gate_thresholds": JAW_STAGE3_GATE,
        "pooled_accuracy": float(pooled_jaw["constrained_exact_accuracy"]),
        "pooled_mae": float(pooled_jaw["constrained_mae"]),
        "summary_rows": jaw_summary_df.to_dict(orient="records"),
    }
    lines = [
        _render_stage_header("Jaw Stage 3: LRJ Jaw Exact-Count Decode", gate_pass),
        f"- LRJ files: `{', '.join(lrj_bundle['contracts'])}`",
        f"- Pooled constrained exact-count accuracy: `{_format_float(float(pooled_jaw['constrained_exact_accuracy']))}`",
        f"- Pooled constrained MAE: `{_format_float(float(pooled_jaw['constrained_mae']))}`",
        "",
        "## Jaw Decode Summary",
        "",
        dataframe_to_markdown(jaw_summary_df, include_index=False),
    ]
    _write_stage_summary(output_dir, "summary", lines, payload)
    return payload


def _run_hand_lrj_stage(
    lrj_bundle: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    decode_df = lrj_bundle["decode_df"]
    summary_df = lrj_bundle["summary_df"]
    hand_decode_df = decode_df[decode_df["label"].isin([LEFT, RIGHT])].copy().reset_index(drop=True)
    hand_summary_df = summary_df[summary_df["label"].isin([LEFT, RIGHT])].copy().reset_index(drop=True)
    hand_decode_df.to_csv(output_dir / "hand_lrj_trial_decode.csv", index=False)
    hand_summary_df.to_csv(output_dir / "hand_lrj_decode_summary.csv", index=False)

    pooled_hand_accuracy = float(hand_decode_df["constrained_exact"].mean()) if not hand_decode_df.empty else 0.0
    pooled_hand_mae = (
        float(np.mean(np.abs(hand_decode_df["constrained_error"].to_numpy(dtype=float))))
        if not hand_decode_df.empty
        else float("nan")
    )
    per_file_rows = []
    for filename, group_df in hand_decode_df.groupby("filename", sort=False):
        per_file_rows.append(
            {
                "filename": filename,
                "trial_count": int(len(group_df)),
                "constrained_exact_accuracy": float(group_df["constrained_exact"].mean()),
                "constrained_mae": float(np.mean(np.abs(group_df["constrained_error"].to_numpy(dtype=float)))),
            }
        )
    per_file_df = pd.DataFrame(per_file_rows).sort_values("filename").reset_index(drop=True)
    per_file_df.to_csv(output_dir / "hand_lrj_file_summary.csv", index=False)
    gate_pass = (
        pooled_hand_accuracy >= EEG_STAGE12_GATE["lrj_exact_accuracy"]
        and pooled_hand_mae <= EEG_STAGE12_GATE["lrj_mae"]
    )
    payload = {
        "gate_pass": gate_pass,
        "gate_thresholds": {
            "exact_accuracy": EEG_STAGE12_GATE["lrj_exact_accuracy"],
            "mae": EEG_STAGE12_GATE["lrj_mae"],
        },
        "pooled_accuracy": pooled_hand_accuracy,
        "pooled_mae": pooled_hand_mae,
        "summary_rows": hand_summary_df.to_dict(orient="records"),
        "per_file_rows": per_file_df.to_dict(orient="records"),
    }
    lines = [
        _render_stage_header("Hand Branch: LRJ LEFT/RIGHT Exact-Count Decode", gate_pass),
        f"- LRJ files: `{', '.join(lrj_bundle['contracts'])}`",
        f"- Pooled constrained exact-count accuracy: `{_format_float(pooled_hand_accuracy)}`",
        f"- Pooled constrained MAE: `{_format_float(pooled_hand_mae)}`",
        "",
        "## Hand Per-File Summary",
        "",
        dataframe_to_markdown(per_file_df, include_index=False),
        "",
        "## Hand Decode By Label",
        "",
        dataframe_to_markdown(hand_summary_df, include_index=False),
    ]
    _write_stage_summary(output_dir, "summary", lines, payload)
    return payload


def _run_eeg_validation_subset(
    contracts: Sequence[FileContract],
    output_dir: Path,
    fs_hz: float,
    make_plot: bool,
    subset_name: str,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    all_block_frames: list[pd.DataFrame] = []
    all_event_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    per_file: dict[str, dict[str, Any]] = {}

    for contract in contracts:
        csv_path = contract.csv_path.resolve()
        raw_df = load_openbci_csv(csv_path)
        audit = audit_session(csv_path, "left_right")
        sampling_note = estimate_sampling(raw_df)
        channel_quality_df = compute_channel_quality(raw_df)
        count_channels = _target_count_channels(channel_quality_df)
        marker_events = collapse_marker_events(raw_df["Marker"]) if "Marker" in raw_df.columns else []
        block_df, pairing_issues = _build_block_table(marker_events, fs_hz)
        file_dir = ensure_output_dir(output_dir / csv_path.stem)

        if block_df.empty:
            run_notes = {
                "file": csv_path.name,
                "issues": ["No valid LEFT/RIGHT marker pairs were found.", *audit.get("issues", [])],
                "pairing_issues": pairing_issues,
                "sampling_note": sampling_note,
                "count_channels": count_channels,
            }
            write_json(file_dir / "run_notes.json", run_notes)
            continue

        count_signal, aggregate_rms_smooth = FROZEN_LR6._build_count_signal(raw_df, count_channels, fs_hz)
        block_summary_df, event_table_df = _build_file_outputs(
            csv_path=csv_path,
            raw_df=raw_df,
            blocks_df=block_df,
            count_signal=count_signal,
            aggregate_rms_smooth=aggregate_rms_smooth,
            fs_hz=fs_hz,
        )
        block_summary_df.to_csv(file_dir / "block_summary.csv", index=False)
        event_table_df.to_csv(file_dir / "event_table.csv", index=False)

        plot_warning = None
        if make_plot:
            plot_warning = _write_plot(
                file_dir / f"{csv_path.stem}_validation.png",
                raw_df=raw_df,
                blocks_df=block_df,
                aggregate_rms_smooth=aggregate_rms_smooth,
                event_table_df=event_table_df,
                fs_hz=fs_hz,
            )

        file_issues = list(dict.fromkeys([*audit.get("issues", []), *pairing_issues]))
        write_json(
            file_dir / "run_notes.json",
            {
                "file": csv_path.name,
                "sampling_note": sampling_note,
                "count_channels": count_channels,
                "channel_quality": channel_quality_df.to_dict(orient="records"),
                "pairing_issues": pairing_issues,
                "audit_issues": audit.get("issues", []),
                "issues": file_issues,
                "plot_warning": plot_warning,
            },
        )

        summary_row, _ = _summarize_file(
            csv_path=csv_path,
            output_dir=file_dir,
            block_summary_df=block_summary_df,
            event_table_df=event_table_df,
            channel_quality_df=channel_quality_df,
            count_channels=count_channels,
            sampling_note=sampling_note,
            issues=file_issues,
            plot_warning=plot_warning,
        )
        summary_row["trust_tier"] = contract.trust_tier
        summary_rows.append(summary_row)
        all_block_frames.append(block_summary_df)
        all_event_frames.append(event_table_df)
        per_file[csv_path.name] = {
            "contract": contract,
            "block_summary_df": block_summary_df,
            "event_table_df": event_table_df,
        }

    summary_df = pd.DataFrame(summary_rows).sort_values("file").reset_index(drop=True)
    all_block_df = pd.concat(all_block_frames, ignore_index=True) if all_block_frames else pd.DataFrame()
    all_event_df = pd.concat(all_event_frames, ignore_index=True) if all_event_frames else pd.DataFrame()
    summary_df.to_csv(output_dir / "overall_summary.csv", index=False)
    all_block_df.to_csv(output_dir / "all_block_summary.csv", index=False)
    all_event_df.to_csv(output_dir / "all_event_table.csv", index=False)

    kept_events = all_event_df[all_event_df["kept_for_count"] == True].copy()
    inside = int((kept_events["inside_marker_block"] == "yes").sum()) if not kept_events.empty else 0
    outside = int((kept_events["inside_marker_block"] == "no").sum()) if not kept_events.empty else 0
    total = inside + outside
    overall_fraction = float(inside / total) if total else float("nan")
    payload = {
        "subset_name": subset_name,
        "summary_rows": summary_df.to_dict(orient="records"),
        "overall_inside_fraction": overall_fraction,
        "kept_events_inside_blocks": inside,
        "kept_events_outside_blocks": outside,
    }
    lines = [
        f"# EEG Stage 1/2 Validation: {subset_name}",
        "",
        f"- Files: `{', '.join(contract.filename for contract in contracts)}`",
        f"- Kept events inside blocks: `{inside}`",
        f"- Kept events outside blocks: `{outside}`",
        f"- Overall inside-block fraction: `{_format_float(overall_fraction)}`",
        "",
        dataframe_to_markdown(summary_df, include_index=False) if not summary_df.empty else "No files processed.",
    ]
    _write_stage_summary(output_dir, "summary", lines, payload)
    payload["summary_df"] = summary_df
    payload["all_event_df"] = all_event_df
    payload["per_file"] = per_file
    return payload


def _compute_common_eeg_channels(contracts: Sequence[FileContract]) -> list[str]:
    common_sets: list[set[str]] = []
    for contract in contracts:
        raw_df = load_openbci_csv(contract.csv_path.resolve())
        quality_df = compute_channel_quality(raw_df)
        common_sets.append(set(_target_count_channels(quality_df)))
    common = sorted(set.intersection(*common_sets), key=_channel_index)
    if len(common) < 2:
        raise ValueError("Fewer than two common EEG channels survived across the main LR files.")
    return common


def _build_lr_event_feature_table(
    contracts: Sequence[FileContract],
    validation_bundle: dict[str, Any],
    common_channels: Sequence[str],
    fs_hz: float,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    window_samples = max(3, int(round(EVENT_WINDOW_SEC * fs_hz)))
    for contract in contracts:
        file_bundle = validation_bundle["per_file"][contract.filename]
        block_lookup = {
            int(row["block_id"]): row for row in file_bundle["block_summary_df"].to_dict(orient="records")
        }
        event_df = file_bundle["event_table_df"][
            (file_bundle["event_table_df"]["kept_for_count"] == True)
            & (file_bundle["event_table_df"]["inside_marker_block"] == "yes")
            & (file_bundle["event_table_df"]["assigned_side"].isin(EEG_SIDE_LABELS))
        ].copy()
        raw_df = load_openbci_csv(contract.csv_path.resolve())
        processed = preprocess_session_signals(raw_df, "left_right", common_channels, fs_hz)
        for event in event_df.to_dict(orient="records"):
            block = block_lookup[int(event["assigned_block_id"])]
            peak_sample = int(round(float(event["event_time_sec"]) * fs_hz))
            block_start_sample = int(round(float(block["start_time_sec"]) * fs_hz))
            block_end_sample = int(round(float(block["end_time_sec"]) * fs_hz))
            window_start, window_end = _window_bounds(
                peak_sample=peak_sample,
                block_start_sample=block_start_sample,
                block_end_sample=block_end_sample,
                window_samples=window_samples,
            )
            window = processed.iloc[window_start : window_end + 1].to_numpy(dtype=float)
            row = {
                "filename": contract.filename,
                "file_path": str(contract.csv_path.resolve()),
                "trust_tier": contract.trust_tier,
                "event_id": int(event["event_id"]),
                "event_time_sec": float(event["event_time_sec"]),
                "event_time_relative_sec": float(event["event_time_relative_sec"]),
                "block_id": int(event["assigned_block_id"]),
                "label": str(event["assigned_side"]),
                "block_start_time_sec": float(block["start_time_sec"]),
                "block_end_time_sec": float(block["end_time_sec"]),
                "peak_value": float(event["peak_value"]),
                "peak_prominence": float(event["peak_prominence"]),
                "window_start_time_sec": float(window_start / fs_hz),
                "window_end_time_sec": float(window_end / fs_hz),
                "window_sec": float((window_end - window_start + 1) / fs_hz),
            }
            row.update(_event_features(window, list(common_channels)))
            rows.append(row)
    feature_df = pd.DataFrame(rows).sort_values(["filename", "event_time_sec"]).reset_index(drop=True)
    metadata_columns = {
        "filename",
        "file_path",
        "trust_tier",
        "event_id",
        "event_time_sec",
        "event_time_relative_sec",
        "block_id",
        "label",
        "block_start_time_sec",
        "block_end_time_sec",
        "peak_value",
        "peak_prominence",
        "window_start_time_sec",
        "window_end_time_sec",
        "window_sec",
    }
    feature_columns = [column for column in feature_df.columns if column not in metadata_columns]
    return feature_df, feature_columns


def _run_eeg_stage3(
    main_contracts: Sequence[FileContract],
    stress_contracts: Sequence[FileContract],
    main_validation: dict[str, Any],
    stress_validation: dict[str, Any],
    output_dir: Path,
    fs_hz: float,
) -> dict[str, Any]:
    output_dir = ensure_output_dir(output_dir)
    common_channels = _compute_common_eeg_channels(main_contracts)
    main_features_df, feature_columns = _build_lr_event_feature_table(main_contracts, main_validation, common_channels, fs_hz)
    stress_features_df, _ = _build_lr_event_feature_table(stress_contracts, stress_validation, common_channels, fs_hz)
    main_features_df.to_csv(output_dir / "main_event_feature_table.csv", index=False)
    stress_features_df.to_csv(output_dir / "stress_event_feature_table.csv", index=False)

    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for test_file in sorted(main_features_df["filename"].unique()):
        train_df = main_features_df[main_features_df["filename"] != test_file].reset_index(drop=True)
        test_df = main_features_df[main_features_df["filename"] == test_file].reset_index(drop=True)
        X_train = train_df[feature_columns].to_numpy(dtype=float)
        y_train = train_df["label"].to_numpy()
        X_test = test_df[feature_columns].to_numpy(dtype=float)
        y_test = test_df["label"].to_numpy()
        for model_name, model in _simple_classifier_bank().items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            confusion_path = output_dir / f"confusion_{train_df['filename'].iloc[0]}_to_{test_file}_{model_name}.csv"
            cm = confusion_matrix(y_test, y_pred, labels=EEG_SIDE_LABELS)
            pd.DataFrame(
                cm,
                index=[f"true_{label}" for label in EEG_SIDE_LABELS],
                columns=[f"pred_{label}" for label in EEG_SIDE_LABELS],
            ).to_csv(confusion_path)
            result_rows.append(
                {
                    "scope": "main_fold",
                    "model_name": model_name,
                    "train_files": ", ".join(sorted(train_df["filename"].unique())),
                    "test_file": test_file,
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "macro_f1": float(f1_score(y_test, y_pred, labels=EEG_SIDE_LABELS, average="macro", zero_division=0)),
                    "confusion_matrix_csv": str(confusion_path.resolve()),
                }
            )
            prediction_frames.append(
                test_df.loc[:, ["filename", "event_id", "label"]]
                .rename(columns={"label": "true_label"})
                .assign(model_name=model_name, pred_label=y_pred, scope="main_fold")
            )

    main_prediction_df = pd.concat(prediction_frames, ignore_index=True)
    pooled_rows: list[dict[str, Any]] = []
    for model_name, model_predictions in main_prediction_df.groupby("model_name"):
        cm = confusion_matrix(model_predictions["true_label"], model_predictions["pred_label"], labels=EEG_SIDE_LABELS)
        confusion_path = output_dir / f"confusion_pooled_{model_name}.csv"
        pd.DataFrame(
            cm,
            index=[f"true_{label}" for label in EEG_SIDE_LABELS],
            columns=[f"pred_{label}" for label in EEG_SIDE_LABELS],
        ).to_csv(confusion_path)
        per_class_recall = {}
        for index, label in enumerate(EEG_SIDE_LABELS):
            denom = float(cm[index].sum())
            per_class_recall[label] = float(cm[index, index] / denom) if denom else 0.0
        pooled_rows.append(
            {
                "scope": "main_pooled",
                "model_name": model_name,
                "train_files": ", ".join(sorted(main_features_df["filename"].unique())),
                "test_file": "__pooled__",
                "accuracy": float(accuracy_score(model_predictions["true_label"], model_predictions["pred_label"])),
                "macro_f1": float(f1_score(model_predictions["true_label"], model_predictions["pred_label"], labels=EEG_SIDE_LABELS, average="macro", zero_division=0)),
                "per_class_recall": json.dumps(per_class_recall),
                "confusion_matrix_csv": str(confusion_path.resolve()),
            }
        )

    stress_rows: list[dict[str, Any]] = []
    if not stress_features_df.empty:
        X_train_main = main_features_df[feature_columns].to_numpy(dtype=float)
        y_train_main = main_features_df["label"].to_numpy()
        for stress_file in sorted(stress_features_df["filename"].unique()):
            test_df = stress_features_df[stress_features_df["filename"] == stress_file].reset_index(drop=True)
            X_test = test_df[feature_columns].to_numpy(dtype=float)
            y_test = test_df["label"].to_numpy()
            for model_name, model in _simple_classifier_bank().items():
                model.fit(X_train_main, y_train_main)
                y_pred = model.predict(X_test)
                confusion_path = output_dir / f"confusion_main_to_{stress_file}_{model_name}.csv"
                pd.DataFrame(
                    confusion_matrix(y_test, y_pred, labels=EEG_SIDE_LABELS),
                    index=[f"true_{label}" for label in EEG_SIDE_LABELS],
                    columns=[f"pred_{label}" for label in EEG_SIDE_LABELS],
                ).to_csv(confusion_path)
                stress_rows.append(
                    {
                        "scope": "stress",
                        "model_name": model_name,
                        "train_files": ", ".join(sorted(main_features_df["filename"].unique())),
                        "test_file": stress_file,
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "macro_f1": float(f1_score(y_test, y_pred, labels=EEG_SIDE_LABELS, average="macro", zero_division=0)),
                        "confusion_matrix_csv": str(confusion_path.resolve()),
                    }
                )

    results_df = pd.DataFrame(result_rows + pooled_rows + stress_rows)
    results_df.to_csv(output_dir / "eeg_stage3_results.csv", index=False)
    main_prediction_df.to_csv(output_dir / "eeg_stage3_predictions.csv", index=False)
    pooled_df = results_df[results_df["scope"] == "main_pooled"].copy().sort_values(["macro_f1", "accuracy"], ascending=False)
    best_row = pooled_df.iloc[0].to_dict()
    per_class_recall = json.loads(best_row["per_class_recall"])
    gate_pass = (
        float(best_row["macro_f1"]) >= EEG_STAGE3_GATE["macro_f1"]
        and min(float(per_class_recall[label]) for label in EEG_SIDE_LABELS) >= EEG_STAGE3_GATE["class_recall"]
    )
    payload = {
        "gate_pass": gate_pass,
        "gate_thresholds": EEG_STAGE3_GATE,
        "best_model": best_row,
        "selected_channels": common_channels,
        "result_rows": results_df.to_dict(orient="records"),
    }
    lines = [
        _render_stage_header("EEG Stage 3: LEFT vs RIGHT Classification", gate_pass),
        f"- Main files: `{', '.join(contract.filename for contract in main_contracts)}`",
        f"- Stress files: `{', '.join(contract.filename for contract in stress_contracts)}`",
        f"- Selected channels: `{', '.join(common_channels)}`",
        f"- Best pooled model: `{best_row['model_name']}`",
        f"- Pooled macro-F1: `{_format_float(best_row['macro_f1'])}`",
        f"- Per-class recall: `{per_class_recall}`",
        "",
        dataframe_to_markdown(results_df, include_index=False),
    ]
    _write_stage_summary(output_dir, "summary", lines, payload)
    return payload


def _build_top_level_summary(
    output_dir: Path,
    registry_df: pd.DataFrame,
    jaw_stage1: dict[str, Any],
    jaw_stage2: dict[str, Any] | None,
    jaw_lrj_stage: dict[str, Any],
    hand_lrj_stage: dict[str, Any],
    eeg_main_validation: dict[str, Any],
    eeg_stress_validation: dict[str, Any],
    eeg_stage3: dict[str, Any] | None,
) -> None:
    jaw_best = jaw_stage1["best_strategy"]
    jaw_stage2_status = jaw_stage2["gate_pass"] if jaw_stage2 is not None else False
    eeg_stage12_main = eeg_main_validation["summary_df"].copy()
    per_file_inside_gate = bool(
        (eeg_stage12_main["inside_block_fraction"] >= EEG_STAGE12_GATE["per_file_inside_fraction"]).all()
    ) if not eeg_stage12_main.empty else False
    overall_inside_gate = float(eeg_main_validation["overall_inside_fraction"]) >= EEG_STAGE12_GATE["overall_inside_fraction"]
    lrj_bridge_gate = bool(hand_lrj_stage["gate_pass"])
    eeg_stage12_pass = per_file_inside_gate and overall_inside_gate and lrj_bridge_gate
    summary_payload = {
        "registry_rows": registry_df.to_dict(orient="records"),
        "jaw_stage1_pass": jaw_stage1["gate_pass"],
        "jaw_stage2_pass": jaw_stage2_status,
        "jaw_stage3_pass": jaw_lrj_stage["gate_pass"],
        "hand_lrj_pass": hand_lrj_stage["gate_pass"],
        "eeg_stage12_pass": eeg_stage12_pass,
        "eeg_stage3_pass": eeg_stage3["gate_pass"] if eeg_stage3 is not None else None,
    }
    lines = [
        "# Stepwise Jaw + EEG Benchmark",
        "",
        f"- Registry file count: `{len(registry_df)}`",
        f"- Diagnostic-only files: `{', '.join(contract.filename for contract in diagnostic_contracts())}`",
        "",
        "## Jaw Branch",
        "",
        f"- Stage 1 pass: `{jaw_stage1['gate_pass']}`",
        f"- Best trigger strategy: `{jaw_best['strategy_name']}` with test weighted event-F1 `{_format_float(jaw_best['test_weighted_event_f1'])}`",
        f"- Stage 2 pass: `{jaw_stage2_status}`",
        f"- Stage 3 pass: `{jaw_lrj_stage['gate_pass']}`",
        f"- LRJ jaw pooled constrained exact-count accuracy: `{_format_float(jaw_lrj_stage['pooled_accuracy'])}`",
        f"- LRJ jaw pooled constrained MAE: `{_format_float(jaw_lrj_stage['pooled_mae'])}`",
        "",
        "## Hand Branch",
        "",
        f"- LRJ hand decode pass: `{hand_lrj_stage['gate_pass']}`",
        f"- LRJ hand pooled constrained exact-count accuracy: `{_format_float(hand_lrj_stage['pooled_accuracy'])}`",
        f"- LRJ hand pooled constrained MAE: `{_format_float(hand_lrj_stage['pooled_mae'])}`",
        f"- Stage 1/2 pass: `{eeg_stage12_pass}`",
        f"- Main inside-block fraction: `{_format_float(eeg_main_validation['overall_inside_fraction'])}`",
        f"- Stress inside-block fraction: `{_format_float(eeg_stress_validation['overall_inside_fraction'])}`",
        f"- Stage 3 pass: `{eeg_stage3['gate_pass'] if eeg_stage3 is not None else False}`",
    ]
    if eeg_stage3 is not None:
        if eeg_stage3.get("skipped"):
            lines.append(f"- EEG Stage 3 status: skipped because `{eeg_stage3['reason']}`")
        else:
            lines.append(
                f"- Best EEG Stage 3 model: `{eeg_stage3['best_model']['model_name']}` with pooled macro-F1 `{_format_float(eeg_stage3['best_model']['macro_f1'])}`"
            )
    lines.extend(
        [
            "",
            "## Registry Preview",
            "",
            dataframe_to_markdown(registry_df, include_index=False),
        ]
    )
    _write_stage_summary(output_dir, "summary", lines, summary_payload)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    registry_df = _validate_registry(output_dir)

    jaw_stage1 = _run_jaw_stage1(jaw_hr_contracts(), output_dir / "jaw_stage1")
    jaw_stage2 = None
    if jaw_stage1["gate_pass"]:
        jaw_stage2 = _run_jaw_type_stage(jaw_stage1["dataset_bundle"]["event_bundle"], output_dir / "jaw_stage2")

    lrj_bundle = _build_lrj_decode_bundle(lrj_main_contracts(), fs_hz=float(args.fs))
    jaw_lrj_stage = _run_jaw_lrj_stage(lrj_bundle, output_dir / "jaw_stage3_lrj")
    hand_lrj_stage = _run_hand_lrj_stage(lrj_bundle, output_dir / "hand_lrj_decode")

    eeg_main_validation = _run_eeg_validation_subset(
        eeg_lr_main_contracts(),
        output_dir / "eeg_stage12_main",
        fs_hz=float(args.fs),
        make_plot=not bool(args.no_plot),
        subset_name="main",
    )
    eeg_stress_validation = _run_eeg_validation_subset(
        eeg_lr_stress_contracts(),
        output_dir / "eeg_stage12_stress",
        fs_hz=float(args.fs),
        make_plot=not bool(args.no_plot),
        subset_name="stress",
    )

    main_summary_df = eeg_main_validation["summary_df"]
    per_file_inside_gate = bool(
        (main_summary_df["inside_block_fraction"] >= EEG_STAGE12_GATE["per_file_inside_fraction"]).all()
    ) if not main_summary_df.empty else False
    overall_inside_gate = float(eeg_main_validation["overall_inside_fraction"]) >= EEG_STAGE12_GATE["overall_inside_fraction"]
    lrj_bridge_gate = bool(hand_lrj_stage["gate_pass"])
    eeg_stage3 = None
    if per_file_inside_gate and overall_inside_gate and lrj_bridge_gate:
        eeg_stage3 = _run_eeg_stage3(
            main_contracts=eeg_lr_main_contracts(),
            stress_contracts=eeg_lr_stress_contracts(),
            main_validation=eeg_main_validation,
            stress_validation=eeg_stress_validation,
            output_dir=output_dir / "eeg_stage3",
            fs_hz=float(args.fs),
        )
    else:
        gate_reasons = []
        if not per_file_inside_gate:
            gate_reasons.append("one or more main LR files stayed below the per-file inside-block fraction gate")
        if not overall_inside_gate:
            gate_reasons.append("the pooled main LR inside-block fraction stayed below the overall gate")
        if not lrj_bridge_gate:
            gate_reasons.append("the hand LRJ constrained decode did not clear its exact-count/MAE gate")
        eeg_stage3 = _write_skipped_stage(
            output_dir / "eeg_stage3",
            "EEG Stage 3: LEFT vs RIGHT Classification",
            "; ".join(gate_reasons),
        )

    _build_top_level_summary(
        output_dir=output_dir,
        registry_df=registry_df,
        jaw_stage1=jaw_stage1,
        jaw_stage2=jaw_stage2,
        jaw_lrj_stage=jaw_lrj_stage,
        hand_lrj_stage=hand_lrj_stage,
        eeg_main_validation=eeg_main_validation,
        eeg_stress_validation=eeg_stress_validation,
        eeg_stage3=eeg_stage3,
    )

    print("Stepwise jaw + EEG benchmark")
    print(f"  Output directory: {output_dir}")
    print(f"  Registry files: {len(registry_df)}")
    print(f"  Jaw Stage 1 pass: {jaw_stage1['gate_pass']}")
    print(f"  Jaw Stage 2 pass: {jaw_stage2['gate_pass'] if jaw_stage2 is not None else False}")
    print(f"  Jaw Stage 3 pass: {jaw_lrj_stage['gate_pass']}")
    print(f"  Hand LRJ decode pass: {hand_lrj_stage['gate_pass']}")
    print(f"  EEG Stage 1/2 main inside-block fraction: {_format_float(eeg_main_validation['overall_inside_fraction'])}")
    print(f"  EEG Stage 1/2 stress inside-block fraction: {_format_float(eeg_stress_validation['overall_inside_fraction'])}")
    print(f"  EEG Stage 3 pass: {eeg_stage3['gate_pass'] if eeg_stage3 is not None else False}")


if __name__ == "__main__":
    main()
