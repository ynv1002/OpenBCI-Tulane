from __future__ import annotations

import argparse
from pathlib import Path
import pickle
from types import SimpleNamespace
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.bci_game_runtime import BCITrackingGameConfig
    from analysis.bci_tracking_game import run_headless_replay_smoke
    from analysis.event_utils import EventWindowConfig, JawEventConfig
    from analysis.jaw_trigger_rules import JawClickTriggerConfig
    from analysis.live_jaw_click_detector import (
        ReplayEvaluationConfig,
        build_jaw_replay_dataset,
        build_replay_score_frame,
        evaluate_clicks_against_reference,
        extract_reference_onsets,
        run_trigger_strategy,
    )
    from analysis.realtime_clench_detector import _build_replay_steps
    from analysis.stepwise_protocol_registry import jaw_hr_contracts, lrj_main_contracts
    from analysis.utils import PROJECT_ROOT, audit_all_sessions, ensure_output_dir
else:
    from .bci_game_runtime import BCITrackingGameConfig
    from .bci_tracking_game import run_headless_replay_smoke
    from .event_utils import EventWindowConfig, JawEventConfig
    from .jaw_trigger_rules import JawClickTriggerConfig
    from .live_jaw_click_detector import (
        ReplayEvaluationConfig,
        build_jaw_replay_dataset,
        build_replay_score_frame,
        evaluate_clicks_against_reference,
        extract_reference_onsets,
        run_trigger_strategy,
    )
    from .realtime_clench_detector import _build_replay_steps
    from .stepwise_protocol_registry import jaw_hr_contracts, lrj_main_contracts
    from .utils import PROJECT_ROOT, audit_all_sessions, ensure_output_dir


DEFAULT_ARTIFACT = PROJECT_ROOT / "models" / "realtime_clench_model.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "outputs" / "jaw_fast_trigger_sweep" / "all_dataset_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the promoted fast jaw model across relevant jaw datasets.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-game-replay", action="store_true")
    return parser.parse_args()


def load_artifact(path: Path) -> dict[str, Any]:
    with path.resolve().open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or "model_bundle" not in artifact:
        raise RuntimeError(f"Unsupported artifact: {path}")
    return artifact


def trigger_from_artifact(artifact: dict[str, Any]) -> JawClickTriggerConfig:
    return JawClickTriggerConfig(**dict(artifact["trigger_config"]))


def evaluate_hr_sessions(artifact: dict[str, Any], output_dir: Path) -> pd.DataFrame:
    window_config = EventWindowConfig(**dict(artifact["window_config"]))
    event_config = JawEventConfig(**dict(artifact["event_config"]))
    dataset_bundle = build_jaw_replay_dataset(audit_all_sessions(), event_config, window_config)
    score_frame = build_replay_score_frame(dataset_bundle, artifact["model_bundle"])
    reference_onsets = extract_reference_onsets(dataset_bundle["event_bundle"]["interval_frame"])
    click_frame = run_trigger_strategy(score_frame, trigger_from_artifact(artifact))
    summary_df, match_df = evaluate_clicks_against_reference(
        click_frame,
        reference_onsets,
        ReplayEvaluationConfig(),
    )
    summary_df.to_csv(output_dir / "hr_jaw_event_eval_by_session.csv", index=False)
    match_df.to_csv(output_dir / "hr_jaw_event_matches.csv", index=False)
    click_frame.to_csv(output_dir / "hr_jaw_click_events.csv", index=False)
    return summary_df


def direct_replay_counts(artifact_path: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for contract in [*jaw_hr_contracts(), *lrj_main_contracts()]:
        args = SimpleNamespace(
            artifact=artifact_path.resolve(),
            csv=contract.csv_path.resolve(),
            threshold=None,
            rearm_threshold=None,
            cooldown_sec=float(load_artifact(artifact_path)["trigger_config"]["cooldown_ms"]) / 1000.0,
        )
        steps, trigger_config, step_interval_sec = _build_replay_steps(args)
        detections = [step for step in steps if step.emitted_click]
        rows.append(
            {
                "key": contract.key,
                "protocol_family": contract.protocol_family,
                "csv_path": str(contract.csv_path.resolve()),
                "step_interval_sec": step_interval_sec,
                "trigger_threshold": trigger_config.clench_probability_threshold,
                "rearm_threshold": trigger_config.rearm_clench_probability_threshold,
                "cooldown_ms": trigger_config.cooldown_ms,
                "replay_steps": len(steps),
                "detected_clicks": len(detections),
                "first_detection_sec": detections[0].timestamp_sec if detections else "",
                "last_detection_sec": detections[-1].timestamp_sec if detections else "",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "direct_replay_click_counts.csv", index=False)
    return frame


def lrj_game_replays(artifact_path: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config = BCITrackingGameConfig(jaw_artifact_path=artifact_path.resolve())
    for contract in lrj_main_contracts():
        result = run_headless_replay_smoke(
            config=config,
            replay_csv=contract.csv_path.resolve(),
            replay_speed=1.0,
            calibration_sec=config.calibration_sec,
        )
        run_dir = Path(result["paths"]["metadata_path"]).parent
        events = pd.read_csv(run_dir / "control_events.csv")
        counts = events["action"].value_counts().to_dict() if not events.empty else {}
        rows.append(
            {
                "key": contract.key,
                "csv_path": str(contract.csv_path.resolve()),
                "run_dir": str(run_dir),
                "clicks": int(counts.get("click", 0)),
                "hold_starts": int(counts.get("hold_start", 0)),
                "hold_ends": int(counts.get("hold_end", 0)),
                "left_events": int(counts.get("LEFT", 0)),
                "right_events": int(counts.get("RIGHT", 0)),
                "final_score": int(result["final_score"]),
                "final_hits": int(result["final_hits"]),
                "final_misses": int(result["final_misses"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "lrj_game_replay_summary.csv", index=False)
    return frame


def write_report(output_dir: Path, hr_summary: pd.DataFrame, direct_counts: pd.DataFrame, lrj_summary: pd.DataFrame | None) -> None:
    weighted = {}
    for split_role, frame in hr_summary.groupby("split_role"):
        weights = frame["reference_onsets"].astype(float)
        weighted[str(split_role)] = {
            "references": int(frame["reference_onsets"].sum()),
            "clicks": int(frame["detected_clicks"].sum()),
            "extra_clicks": int(frame["extra_clicks"].sum()),
            "precision": float((frame["precision"] * weights).sum() / weights.sum()),
            "recall": float((frame["recall"] * weights).sum() / weights.sum()),
            "event_f1": float((frame["event_f1"] * weights).sum() / weights.sum()),
        }

    lines = [
        "# Fast Jaw Model All-Dataset Validation",
        "",
        "## HR Jaw Event Evaluation",
        "",
        hr_summary[
            [
                "filename",
                "split_role",
                "reference_onsets",
                "detected_clicks",
                "matched_clicks",
                "extra_clicks",
                "precision",
                "recall",
                "event_f1",
                "median_lag_ms",
            ]
        ].to_markdown(index=False),
        "",
        "## Weighted HR Summary",
        "",
        pd.DataFrame.from_dict(weighted, orient="index").reset_index(names="split_role").to_markdown(index=False),
        "",
        "## Direct Replay Click Counts",
        "",
        direct_counts[["key", "protocol_family", "detected_clicks", "first_detection_sec", "last_detection_sec"]].to_markdown(index=False),
    ]
    if lrj_summary is not None:
        lines.extend(
            [
                "",
                "## LRJ Game Replay Output",
                "",
                lrj_summary[
                    [
                        "key",
                        "clicks",
                        "hold_starts",
                        "hold_ends",
                        "left_events",
                        "right_events",
                        "final_score",
                        "final_hits",
                        "final_misses",
                        "run_dir",
                    ]
                ].to_markdown(index=False),
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir.resolve())
    artifact_path = args.artifact.resolve()
    artifact = load_artifact(artifact_path)
    hr_summary = evaluate_hr_sessions(artifact, output_dir)
    direct_counts = direct_replay_counts(artifact_path, output_dir)
    lrj_summary = None if args.skip_game_replay else lrj_game_replays(artifact_path, output_dir)
    write_report(output_dir, hr_summary, direct_counts, lrj_summary)
    print(f"Wrote all-dataset validation to {output_dir}")


if __name__ == "__main__":
    main()
