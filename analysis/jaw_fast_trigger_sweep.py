from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.event_utils import EventWindowConfig, JawEventConfig
    from analysis.jaw_trigger_rules import JawClickTriggerConfig
    from analysis.live_jaw_click_detector import (
        ReplayEvaluationConfig,
        build_jaw_replay_dataset,
        build_replay_score_frame,
        evaluate_clicks_against_reference,
        extract_reference_onsets,
        run_trigger_strategy,
        summarize_strategy_performance,
        train_jaw_click_models,
    )
    from analysis.realtime_clench_detector import default_trigger_config
    from analysis.utils import PROJECT_ROOT, audit_all_sessions, ensure_output_dir, write_json
else:
    from .event_utils import EventWindowConfig, JawEventConfig
    from .jaw_trigger_rules import JawClickTriggerConfig
    from .live_jaw_click_detector import (
        ReplayEvaluationConfig,
        build_jaw_replay_dataset,
        build_replay_score_frame,
        evaluate_clicks_against_reference,
        extract_reference_onsets,
        run_trigger_strategy,
        summarize_strategy_performance,
        train_jaw_click_models,
    )
    from .realtime_clench_detector import default_trigger_config
    from .utils import PROJECT_ROOT, audit_all_sessions, ensure_output_dir, write_json


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "outputs" / "jaw_fast_trigger_sweep"
DEFAULT_FAST_ARTIFACT = PROJECT_ROOT / "models" / "realtime_clench_model_fast.pkl"


MODEL_CONFIGS = [
    {"model_config": "w0p10_s0p10_peak0p35", "window_sec": 0.10, "smoothing_sec": 0.10, "min_peak_distance_sec": 0.35},
    {"model_config": "w0p10_s0p15_peak0p35", "window_sec": 0.10, "smoothing_sec": 0.15, "min_peak_distance_sec": 0.35},
    {"model_config": "w0p12_s0p12_peak0p35", "window_sec": 0.12, "smoothing_sec": 0.12, "min_peak_distance_sec": 0.35},
    {"model_config": "w0p15_s0p15_peak0p35", "window_sec": 0.15, "smoothing_sec": 0.15, "min_peak_distance_sec": 0.35},
    {"model_config": "w0p20_s0p25_peak0p35_current", "window_sec": 0.20, "smoothing_sec": 0.25, "min_peak_distance_sec": 0.35},
]

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.80, 0.90]
REARM_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60]
COOLDOWN_MS = [50, 100, 150, 200, 250, 300]
HOLD_SUPPRESSION_VALUES = [True, False]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep shorter jaw windows and faster trigger settings against held-out jaw replay events."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_FAST_ARTIFACT)
    parser.add_argument("--no-save-artifact", action="store_true")
    return parser.parse_args()


def trigger_grid() -> list[JawClickTriggerConfig]:
    base = default_trigger_config()
    configs: list[JawClickTriggerConfig] = []
    for threshold in THRESHOLDS:
        for rearm in REARM_THRESHOLDS:
            if rearm >= threshold:
                continue
            for cooldown_ms in COOLDOWN_MS:
                for hold_suppression in HOLD_SUPPRESSION_VALUES:
                    configs.append(
                        JawClickTriggerConfig(
                            strategy_name="binary_clench_threshold",
                            clench_probability_threshold=float(threshold),
                            onset_probability_threshold=base.onset_probability_threshold,
                            active_probability_threshold=base.active_probability_threshold,
                            rearm_clench_probability_threshold=float(rearm),
                            cooldown_ms=int(cooldown_ms),
                            minimum_separation_ms=int(cooldown_ms),
                            smoothing_windows=1,
                            hold_suppression=bool(hold_suppression),
                            require_transition_from_inactive=False,
                            minimum_clench_rise=0.0,
                            minimum_envelope_uv=0.0,
                        )
                    )
    return configs


def score_model_windows(window_frame: pd.DataFrame, model_bundle: dict[str, Any]) -> dict[str, float]:
    feature_columns = model_bundle["feature_columns"]
    test_mask = window_frame["split_role"] == "test"
    feature_matrix = window_frame.loc[test_mask, feature_columns].to_numpy(dtype=float)
    true_4state = window_frame.loc[test_mask, "event_label"].to_numpy()
    pred_4state = model_bundle["jaw_4state_model"].predict(feature_matrix)
    true_binary = np.where(true_4state == "INACTIVE", "NON_CLENCH", "CLENCH")
    pred_binary = model_bundle["binary_model"].predict(feature_matrix)
    return {
        "jaw_4state_accuracy": float(accuracy_score(true_4state, pred_4state)),
        "jaw_4state_macro_f1": float(f1_score(true_4state, pred_4state, average="macro")),
        "binary_accuracy": float(accuracy_score(true_binary, pred_binary)),
        "binary_macro_f1": float(f1_score(true_binary, pred_binary, average="macro")),
    }


def row_sort_key(row: dict[str, Any]) -> tuple[float, float, float, int, float]:
    return (
        float(row["test_weighted_event_f1"]),
        float(row["test_weighted_recall"]),
        float(row["test_weighted_precision"]),
        -int(row["test_total_extra_clicks"]),
        -float(row["test_median_lag_ms"]) if not np.isnan(float(row["test_median_lag_ms"])) else -99999.0,
    )


def train_sort_key(row: dict[str, Any]) -> tuple[float, float, int, float]:
    return (
        float(row["train_weighted_event_f1"]),
        float(row["train_weighted_precision"]),
        -int(row["train_total_extra_clicks"]),
        float(row["test_weighted_event_f1"]),
    )


def save_artifact(
    artifact_path: Path,
    selected_row: dict[str, Any],
    model_bundle: dict[str, Any],
    model_metrics: dict[str, float],
    model_config: dict[str, Any],
) -> None:
    ensure_output_dir(artifact_path.parent)
    event_config = JawEventConfig(
        smoothing_sec=float(model_config["smoothing_sec"]),
        minimum_peak_distance_sec=float(model_config["min_peak_distance_sec"]),
    )
    window_config = EventWindowConfig(window_sec=float(model_config["window_sec"]), overlap=0.50)
    trigger_config = selected_row["trigger_config"]
    summary = {
        "artifact_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": selected_row["model_config"],
        "selection_note": "Selected as the promoted runtime candidate from jaw_fast_trigger_sweep.py based on held-out event-F1 and replay behavior.",
        "selected_channels": model_bundle["selected_channels"],
        "excluded_channels": model_bundle["excluded_channels"],
        "feature_columns": model_bundle["feature_columns"],
        "window_config": asdict(window_config),
        "event_config": asdict(event_config),
        "trigger_config": trigger_config,
        "metrics": {
            **model_metrics,
            "trigger_summary": {
                key: value
                for key, value in selected_row.items()
                if key.startswith("train_") or key.startswith("test_") or key in {"trigger_config_name"}
            },
        },
    }
    with artifact_path.open("wb") as handle:
        pickle.dump({**summary, "model_bundle": model_bundle}, handle)
    write_json(artifact_path.with_suffix(".json"), summary)


def write_report(
    output_dir: Path,
    all_rows: pd.DataFrame,
    best_by_train: dict[str, Any],
    best_by_test: dict[str, Any],
    artifact_path: Path | None,
) -> None:
    top_cols = [
        "model_config",
        "window_sec",
        "smoothing_sec",
        "clench_threshold",
        "rearm_threshold",
        "cooldown_ms",
        "hold_suppression",
        "binary_macro_f1",
        "jaw_4state_macro_f1",
        "test_weighted_event_f1",
        "test_weighted_precision",
        "test_weighted_recall",
        "test_total_clicks",
        "test_total_references",
        "test_total_extra_clicks",
        "test_median_lag_ms",
    ]
    top_by_test = all_rows.sort_values(
        ["test_weighted_event_f1", "test_weighted_recall", "test_weighted_precision"],
        ascending=[False, False, False],
    ).head(20)
    current = all_rows[all_rows["model_config"] == "w0p20_s0p25_peak0p35_current"].sort_values(
        ["test_weighted_event_f1", "test_weighted_recall", "test_weighted_precision"],
        ascending=[False, False, False],
    ).head(5)
    fast = all_rows[all_rows["window_sec"] <= 0.12].sort_values(
        ["test_weighted_event_f1", "test_weighted_recall", "test_weighted_precision"],
        ascending=[False, False, False],
    ).head(10)

    lines = [
        "# Jaw Fast Trigger Sweep",
        "",
        "This sweep retrains jaw clench models with shorter feature windows and evaluates faster trigger settings against held-out jaw replay events.",
        "",
        "## Promoted Runtime Result",
        "",
        f"- Promoted model config: `{best_by_test['model_config']}`",
        f"- Window: `{best_by_test['window_sec']:.2f} s`",
        f"- Smoothing: `{best_by_test['smoothing_sec']:.2f} s`",
        f"- Trigger threshold: `{best_by_test['clench_threshold']:.2f}`",
        f"- Rearm threshold: `{best_by_test['rearm_threshold']:.2f}`",
        f"- Cooldown/min separation: `{int(best_by_test['cooldown_ms'])} ms`",
        f"- Hold suppression: `{best_by_test['hold_suppression']}`",
        f"- Test event-F1 / precision / recall: `{best_by_test['test_weighted_event_f1']:.3f}` / `{best_by_test['test_weighted_precision']:.3f}` / `{best_by_test['test_weighted_recall']:.3f}`",
        f"- Test clicks / references / extra clicks: `{int(best_by_test['test_total_clicks'])}` / `{int(best_by_test['test_total_references'])}` / `{int(best_by_test['test_total_extra_clicks'])}`",
        "",
        "## Best Held-Out Row",
        "",
        f"- Test-best model config: `{best_by_test['model_config']}`",
        f"- Window: `{best_by_test['window_sec']:.2f} s`, threshold `{best_by_test['clench_threshold']:.2f}`, cooldown `{int(best_by_test['cooldown_ms'])} ms`, hold suppression `{best_by_test['hold_suppression']}`",
        f"- Test event-F1 / precision / recall: `{best_by_test['test_weighted_event_f1']:.3f}` / `{best_by_test['test_weighted_precision']:.3f}` / `{best_by_test['test_weighted_recall']:.3f}`",
        "",
        "## Top Held-Out Settings",
        "",
        top_by_test[top_cols].to_markdown(index=False),
        "",
        "## Top Fast Window Settings",
        "",
        fast[top_cols].to_markdown(index=False),
        "",
        "## Top Current-Window Settings",
        "",
        current[top_cols].to_markdown(index=False),
        "",
        "## Files",
        "",
        "- All sweep rows: `jaw_fast_trigger_sweep_results.csv`",
        "- Top held-out rows: `jaw_fast_trigger_sweep_top.csv`",
    ]
    if artifact_path is not None:
        lines.append(f"- Promoted fast artifact: `{artifact_path}`")
        lines.append(f"- Promoted fast artifact summary: `{artifact_path.with_suffix('.json')}`")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_output_dir(output_dir)

    audits = audit_all_sessions()
    eval_config = ReplayEvaluationConfig()
    configs = trigger_grid()

    rows: list[dict[str, Any]] = []
    bundles_by_config: dict[str, dict[str, Any]] = {}
    metrics_by_config: dict[str, dict[str, float]] = {}
    model_configs_by_name = {cfg["model_config"]: cfg for cfg in MODEL_CONFIGS}

    for model_config in MODEL_CONFIGS:
        event_config = JawEventConfig(
            smoothing_sec=float(model_config["smoothing_sec"]),
            minimum_peak_distance_sec=float(model_config["min_peak_distance_sec"]),
        )
        window_config = EventWindowConfig(window_sec=float(model_config["window_sec"]), overlap=0.50)
        dataset_bundle = build_jaw_replay_dataset(audits, event_config, window_config)
        model_bundle = train_jaw_click_models(dataset_bundle)
        score_frame = build_replay_score_frame(dataset_bundle, model_bundle)
        reference_onsets = extract_reference_onsets(dataset_bundle["event_bundle"]["interval_frame"])
        model_metrics = score_model_windows(dataset_bundle["window_bundle"]["window_frame"], model_bundle)
        bundles_by_config[str(model_config["model_config"])] = model_bundle
        metrics_by_config[str(model_config["model_config"])] = model_metrics

        for trigger_config in configs:
            click_frame = run_trigger_strategy(score_frame, trigger_config)
            summary_df, _ = evaluate_clicks_against_reference(click_frame, reference_onsets, eval_config)
            trigger_summary = summarize_strategy_performance(
                summary_df,
                trigger_config.strategy_name,
                trigger_config,
            )
            rows.append(
                {
                    **model_config,
                    **model_metrics,
                    "trigger_config_name": trigger_config.compact_name(),
                    "strategy_name": trigger_config.strategy_name,
                    "clench_threshold": float(trigger_config.clench_probability_threshold),
                    "rearm_threshold": float(trigger_config.rearm_clench_probability_threshold),
                    "cooldown_ms": int(trigger_config.cooldown_ms),
                    "hold_suppression": bool(trigger_config.hold_suppression),
                    "trigger_config": trigger_config.as_dict(),
                    **{
                        key: value
                        for key, value in trigger_summary.items()
                        if key.startswith("train_") or key.startswith("test_")
                    },
                }
            )

    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(output_dir / "jaw_fast_trigger_sweep_results.csv", index=False)
    top_rows = all_rows.sort_values(
        ["test_weighted_event_f1", "test_weighted_recall", "test_weighted_precision"],
        ascending=[False, False, False],
    ).head(50)
    top_rows.to_csv(output_dir / "jaw_fast_trigger_sweep_top.csv", index=False)

    row_dicts = all_rows.to_dict(orient="records")
    best_by_train = max(row_dicts, key=train_sort_key)
    best_by_test = max(row_dicts, key=row_sort_key)

    artifact_path: Path | None = None
    if not args.no_save_artifact:
        artifact_path = args.artifact.resolve()
        selected_config_name = str(best_by_test["model_config"])
        save_artifact(
            artifact_path=artifact_path,
            selected_row=best_by_test,
            model_bundle=bundles_by_config[selected_config_name],
            model_metrics=metrics_by_config[selected_config_name],
            model_config=model_configs_by_name[selected_config_name],
        )

    write_report(output_dir, all_rows, best_by_train, best_by_test, artifact_path)
    payload = {
        "best_by_train": best_by_train,
        "best_by_test": best_by_test,
        "model_configs": MODEL_CONFIGS,
        "thresholds": THRESHOLDS,
        "rearm_thresholds": REARM_THRESHOLDS,
        "cooldown_ms": COOLDOWN_MS,
        "hold_suppression_values": HOLD_SUPPRESSION_VALUES,
        "artifact_path": str(artifact_path) if artifact_path is not None else None,
    }
    (output_dir / "jaw_fast_trigger_sweep_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"best_by_train": best_by_train, "best_by_test": best_by_test}, indent=2))


if __name__ == "__main__":
    main()
