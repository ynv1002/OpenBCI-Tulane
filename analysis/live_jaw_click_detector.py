from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .event_utils import (
    EVENT_ACTIVE,
    EVENT_INACTIVE,
    EVENT_ONSET,
    EventWindowConfig,
    JawEventConfig,
    build_event_windows,
    build_jaw_event_label_bundle,
    remap_event_labels,
)
from .jaw_trigger_rules import JawClickTrigger, JawClickTriggerConfig
from .utils import write_json


@dataclass(frozen=True)
class ReplayEvaluationConfig:
    match_pre_sec: float = 0.15
    match_post_sec: float = 0.45


def build_jaw_replay_dataset(
    audits: Sequence[Dict[str, Any]],
    event_config: JawEventConfig,
    window_config: EventWindowConfig,
) -> Dict[str, Any]:
    event_bundle = build_jaw_event_label_bundle(audits, event_config)
    window_bundle = build_event_windows(
        event_bundle["sample_frame"],
        event_bundle["signal_columns"],
        window_config,
    )
    return {
        "event_bundle": event_bundle,
        "window_bundle": window_bundle,
    }


def _logistic_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )


def train_jaw_click_models(dataset_bundle: Dict[str, Any]) -> Dict[str, Any]:
    window_df = dataset_bundle["window_bundle"]["window_frame"]
    feature_columns = dataset_bundle["window_bundle"]["feature_columns"]

    jaw_4state_df = remap_event_labels(window_df, "jaw_4state")
    binary_df = remap_event_labels(window_df, "clench_vs_nonclench")
    train_mask = jaw_4state_df["split_role"] == "train"

    jaw_4state_model = _logistic_pipeline()
    jaw_4state_model.fit(
        jaw_4state_df.loc[train_mask, feature_columns].to_numpy(dtype=float),
        jaw_4state_df.loc[train_mask, "label"].to_numpy(),
    )

    binary_model = _logistic_pipeline()
    binary_model.fit(
        binary_df.loc[train_mask, feature_columns].to_numpy(dtype=float),
        binary_df.loc[train_mask, "label"].to_numpy(),
    )

    return {
        "jaw_4state_model": jaw_4state_model,
        "jaw_4state_labels": list(jaw_4state_model.named_steps["model"].classes_),
        "binary_model": binary_model,
        "binary_labels": list(binary_model.named_steps["model"].classes_),
        "feature_columns": feature_columns,
        "selected_channels": dataset_bundle["event_bundle"]["selected_channels"],
        "excluded_channels": dataset_bundle["event_bundle"]["excluded_channels"],
    }


def build_replay_score_frame(dataset_bundle: Dict[str, Any], model_bundle: Dict[str, Any]) -> pd.DataFrame:
    window_df = dataset_bundle["window_bundle"]["window_frame"].copy()
    feature_columns = model_bundle["feature_columns"]
    X = window_df[feature_columns].to_numpy(dtype=float)

    jaw_4state_probs = model_bundle["jaw_4state_model"].predict_proba(X)
    jaw_4state_labels = list(model_bundle["jaw_4state_labels"])
    binary_probs = model_bundle["binary_model"].predict_proba(X)
    binary_labels = list(model_bundle["binary_labels"])

    for label_index, label_name in enumerate(jaw_4state_labels):
        window_df[f"prob_4state_{label_name}"] = jaw_4state_probs[:, label_index]
    for label_index, label_name in enumerate(binary_labels):
        window_df[f"prob_binary_{label_name}"] = binary_probs[:, label_index]

    window_df["pred_4state"] = model_bundle["jaw_4state_model"].predict(X)
    window_df["pred_binary"] = model_bundle["binary_model"].predict(X)
    window_df["event_label_true"] = window_df["event_label"]

    return window_df


def extract_reference_onsets(interval_frame: pd.DataFrame) -> pd.DataFrame:
    onset_rows = interval_frame[interval_frame["state_label"] == EVENT_ONSET].copy()
    onset_rows = onset_rows.sort_values(["filename", "start_time_sec", "event_id"]).reset_index(drop=True)
    onset_rows["reference_onset_time_sec"] = onset_rows["start_time_sec"]
    onset_rows["reference_kind"] = "derived_event_onset"
    return onset_rows[
        [
            "family",
            "filename",
            "file_path",
            "parsed_date",
            "session_rank",
            "split_role",
            "coarse_label",
            "coarse_segment_index",
            "event_id",
            "event_source",
            "reference_kind",
            "reference_onset_time_sec",
            "start_time_sec",
            "end_time_sec",
            "duration_sec",
        ]
    ].rename(columns={"start_time_sec": "reference_interval_start_sec", "end_time_sec": "reference_interval_end_sec"})


def build_default_trigger_grid() -> List[JawClickTriggerConfig]:
    configs: List[JawClickTriggerConfig] = []
    for threshold in (0.60, 0.70, 0.80):
        for smoothing in (1, 3):
            for cooldown in (300, 450, 600):
                configs.append(
                    JawClickTriggerConfig(
                        strategy_name="binary_clench_threshold",
                        clench_probability_threshold=threshold,
                        onset_probability_threshold=0.45,
                        active_probability_threshold=0.55,
                        rearm_clench_probability_threshold=max(0.20, threshold - 0.30),
                        cooldown_ms=cooldown,
                        minimum_separation_ms=cooldown,
                        smoothing_windows=smoothing,
                        hold_suppression=True,
                        require_transition_from_inactive=False,
                        minimum_clench_rise=0.00,
                    )
                )
    for threshold in (0.35, 0.45, 0.55):
        for smoothing in (1, 3):
            for cooldown in (250, 400, 550):
                configs.append(
                    JawClickTriggerConfig(
                        strategy_name="onset_threshold",
                        clench_probability_threshold=0.70,
                        onset_probability_threshold=threshold,
                        active_probability_threshold=0.55,
                        rearm_clench_probability_threshold=0.30,
                        cooldown_ms=cooldown,
                        minimum_separation_ms=cooldown,
                        smoothing_windows=smoothing,
                        hold_suppression=True,
                        require_transition_from_inactive=False,
                        minimum_clench_rise=0.00,
                    )
                )
    for onset_threshold in (0.35, 0.45):
        for clench_threshold in (0.60, 0.70):
            for smoothing in (1, 3):
                for cooldown in (350, 500):
                    for rise in (0.04, 0.08):
                        configs.append(
                            JawClickTriggerConfig(
                                strategy_name="hybrid_transition",
                                clench_probability_threshold=clench_threshold,
                                onset_probability_threshold=onset_threshold,
                                active_probability_threshold=0.50,
                                rearm_clench_probability_threshold=max(0.20, clench_threshold - 0.30),
                                cooldown_ms=cooldown,
                                minimum_separation_ms=cooldown,
                                smoothing_windows=smoothing,
                                hold_suppression=True,
                                require_transition_from_inactive=True,
                                minimum_clench_rise=rise,
                            )
                        )
    return configs


def run_trigger_strategy(
    score_frame: pd.DataFrame,
    trigger_config: JawClickTriggerConfig,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    detector = JawClickTrigger(trigger_config)

    for filename, session_df in score_frame.groupby("filename", sort=False):
        detector.reset()
        session_df = session_df.sort_values("center_time_sec").reset_index(drop=True)
        for _, row in session_df.iterrows():
            scores = {
                "clench_probability": float(row.get("prob_binary_CLENCH", 0.0)),
                "non_clench_probability": float(row.get("prob_binary_NON_CLENCH", 0.0)),
                "inactive_probability": float(row.get("prob_4state_INACTIVE", 0.0)),
                "onset_probability": float(row.get("prob_4state_ONSET", 0.0)),
                "active_probability": float(row.get("prob_4state_ACTIVE", 0.0)),
                "offset_probability": float(row.get("prob_4state_OFFSET", 0.0)),
                "envelope_uv": float(row.get("aggregate_rms_smooth_mean", 0.0)),
            }
            decision = detector.step(
                timestamp_sec=float(row["center_time_sec"]),
                scores=scores,
                event_label=str(row["pred_4state"]),
            )
            if decision.emitted_click:
                rows.append(
                    {
                        "strategy_name": trigger_config.strategy_name,
                        "trigger_config_name": trigger_config.compact_name(),
                        "filename": filename,
                        "file_path": row["file_path"],
                        "parsed_date": row["parsed_date"],
                        "session_rank": int(row["session_rank"]),
                        "split_role": row["split_role"],
                        "center_time_sec": float(row["center_time_sec"]),
                        "coarse_label_center": row["coarse_label_center"],
                        "pred_4state": row["pred_4state"],
                        "event_label_true": row["event_label_true"],
                        "trigger_reason": decision.reason,
                        "clench_probability": decision.smoothed_scores["clench_probability"],
                        "onset_probability": decision.smoothed_scores["onset_probability"],
                        "active_probability": decision.smoothed_scores["active_probability"],
                        "inactive_probability": decision.smoothed_scores["inactive_probability"],
                        "aggregate_rms_smooth_mean": decision.smoothed_scores["envelope_uv"],
                        "armed_after_step": decision.armed,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_clicks_against_reference(
    click_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    evaluation_config: ReplayEvaluationConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    session_keys = reference_frame[
        ["filename", "file_path", "parsed_date", "session_rank", "split_role"]
    ].drop_duplicates()

    for _, session_key in session_keys.iterrows():
        filename = session_key["filename"]
        ref_df = reference_frame[reference_frame["filename"] == filename].sort_values("reference_onset_time_sec")
        click_df = click_frame[click_frame["filename"] == filename].sort_values("center_time_sec")
        refs = ref_df["reference_onset_time_sec"].to_numpy(dtype=float)
        clicks = click_df["center_time_sec"].to_numpy(dtype=float)

        matched_click_indices = set()
        matched_ref_indices = set()
        lags_sec: List[float] = []
        for ref_index, ref_time in enumerate(refs):
            best_click_index = None
            best_abs_lag = None
            for click_index, click_time in enumerate(clicks):
                if click_index in matched_click_indices:
                    continue
                lag = float(click_time - ref_time)
                if lag < -evaluation_config.match_pre_sec or lag > evaluation_config.match_post_sec:
                    continue
                abs_lag = abs(lag)
                if best_click_index is None or abs_lag < float(best_abs_lag):
                    best_click_index = click_index
                    best_abs_lag = abs_lag
                    best_lag = lag
            if best_click_index is not None:
                matched_click_indices.add(best_click_index)
                matched_ref_indices.add(ref_index)
                lags_sec.append(float(best_lag))
                click_row = click_df.iloc[int(best_click_index)]
                ref_row = ref_df.iloc[int(ref_index)]
                match_rows.append(
                    {
                        "filename": filename,
                        "split_role": session_key["split_role"],
                        "reference_event_id": int(ref_row["event_id"]),
                        "reference_onset_time_sec": float(ref_row["reference_onset_time_sec"]),
                        "click_time_sec": float(click_row["center_time_sec"]),
                        "lag_sec": float(best_lag),
                        "coarse_label": ref_row["coarse_label"],
                        "event_source": ref_row["event_source"],
                        "trigger_reason": click_row["trigger_reason"],
                    }
                )

        matched_clicks = len(matched_click_indices)
        reference_count = len(refs)
        click_count = len(clicks)
        precision = float(matched_clicks / click_count) if click_count else 0.0
        recall = float(len(matched_ref_indices) / reference_count) if reference_count else 0.0
        event_f1 = float(
            2.0 * precision * recall / (precision + recall)
        ) if precision > 0.0 and recall > 0.0 else 0.0

        summary_rows.append(
            {
                "filename": filename,
                "file_path": session_key["file_path"],
                "parsed_date": session_key["parsed_date"],
                "session_rank": int(session_key["session_rank"]),
                "split_role": session_key["split_role"],
                "reference_onsets": reference_count,
                "detected_clicks": click_count,
                "matched_clicks": matched_clicks,
                "missed_onsets": reference_count - len(matched_ref_indices),
                "extra_clicks": click_count - matched_clicks,
                "precision": precision,
                "recall": recall,
                "event_f1": event_f1,
                "median_lag_ms": float(np.median(lags_sec) * 1000.0) if lags_sec else np.nan,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(match_rows)


def summarize_strategy_performance(
    summary_df: pd.DataFrame,
    strategy_name: str,
    trigger_config: JawClickTriggerConfig,
) -> Dict[str, Any]:
    train_df = summary_df[summary_df["split_role"] == "train"]
    test_df = summary_df[summary_df["split_role"] == "test"]

    def _weighted_metric(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
        if frame.empty:
            return 0.0
        weights = frame[weight_col].to_numpy(dtype=float)
        values = frame[value_col].to_numpy(dtype=float)
        if float(weights.sum()) == 0.0:
            return float(values.mean()) if len(values) else 0.0
        return float(np.average(values, weights=weights))

    summary = {
        "strategy_name": strategy_name,
        "trigger_config": asdict(trigger_config),
        "trigger_config_name": trigger_config.compact_name(),
        "train_weighted_precision": _weighted_metric(train_df, "precision", "reference_onsets"),
        "train_weighted_recall": _weighted_metric(train_df, "recall", "reference_onsets"),
        "train_weighted_event_f1": _weighted_metric(train_df, "event_f1", "reference_onsets"),
        "train_total_references": int(train_df["reference_onsets"].sum()) if not train_df.empty else 0,
        "train_total_clicks": int(train_df["detected_clicks"].sum()) if not train_df.empty else 0,
        "train_total_extra_clicks": int(train_df["extra_clicks"].sum()) if not train_df.empty else 0,
        "test_weighted_precision": _weighted_metric(test_df, "precision", "reference_onsets"),
        "test_weighted_recall": _weighted_metric(test_df, "recall", "reference_onsets"),
        "test_weighted_event_f1": _weighted_metric(test_df, "event_f1", "reference_onsets"),
        "test_total_references": int(test_df["reference_onsets"].sum()) if not test_df.empty else 0,
        "test_total_clicks": int(test_df["detected_clicks"].sum()) if not test_df.empty else 0,
        "test_total_extra_clicks": int(test_df["extra_clicks"].sum()) if not test_df.empty else 0,
        "test_median_lag_ms": float(test_df["median_lag_ms"].median()) if not test_df.empty else np.nan,
    }
    return summary


def choose_best_config(train_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return max(
        train_summaries,
        key=lambda item: (
            item["train_weighted_event_f1"],
            item["train_weighted_precision"],
            -item["train_total_extra_clicks"],
        ),
    )


def export_strategy_artifacts(
    output_dir: Path,
    strategy_name: str,
    score_frame: pd.DataFrame,
    click_frame: pd.DataFrame,
    summary_df: pd.DataFrame,
    match_df: pd.DataFrame,
) -> Dict[str, str]:
    score_path = output_dir / "jaw_click_replay_scores.csv"
    if not score_path.exists():
        score_frame.to_csv(score_path, index=False)

    click_path = output_dir / f"jaw_click_events_{strategy_name}.csv"
    summary_path = output_dir / f"jaw_click_eval_{strategy_name}.csv"
    match_path = output_dir / f"jaw_click_matches_{strategy_name}.csv"
    click_frame.to_csv(click_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    match_df.to_csv(match_path, index=False)
    return {
        "score_csv": str(score_path),
        "click_csv": str(click_path),
        "summary_csv": str(summary_path),
        "match_csv": str(match_path),
    }


def write_best_strategy_report(
    path_md: Path,
    path_json: Path,
    best_strategy_summary: Dict[str, Any],
    strategy_rows: pd.DataFrame,
    evaluation_config: ReplayEvaluationConfig,
) -> None:
    payload = {
        "best_strategy": best_strategy_summary,
        "all_strategy_summaries": strategy_rows.to_dict(orient="records"),
        "evaluation_config": asdict(evaluation_config),
        "label_caveat": "Reference onset times come from derived event labels built from approximate marker timing plus signal heuristics. They are useful replay supervision, not exact physiological truth.",
    }
    write_json(path_json, payload)

    lines = [
        "# Jaw Click Replay Summary",
        "",
        "The replay evaluation uses approximate onset neighborhoods from derived event labels.",
        "Marker edges are experimenter timing, not exact physiological onset truth.",
        "",
        f"- Best strategy: `{best_strategy_summary['strategy_name']}`",
        f"- Trigger config: `{best_strategy_summary['trigger_config']}`",
        f"- Train weighted event-F1: `{best_strategy_summary['train_weighted_event_f1']:.3f}`",
        f"- Test weighted event-F1: `{best_strategy_summary['test_weighted_event_f1']:.3f}`",
        f"- Test weighted precision: `{best_strategy_summary['test_weighted_precision']:.3f}`",
        f"- Test weighted recall: `{best_strategy_summary['test_weighted_recall']:.3f}`",
        f"- Test total clicks: `{best_strategy_summary['test_total_clicks']}`",
        f"- Test total approximate onset references: `{best_strategy_summary['test_total_references']}`",
        f"- Test extra clicks: `{best_strategy_summary['test_total_extra_clicks']}`",
        f"- Test median lag (ms): `{best_strategy_summary['test_median_lag_ms']:.1f}`",
        "",
        "## Strategy Comparison",
        "",
        strategy_rows.to_markdown(index=False),
        "",
        "## Evaluation Window",
        "",
        f"- early allowance before approximate onset: `{evaluation_config.match_pre_sec:.2f} s`",
        f"- late allowance after approximate onset: `{evaluation_config.match_post_sec:.2f} s`",
    ]
    path_md.write_text("\n".join(lines), encoding="utf-8")
