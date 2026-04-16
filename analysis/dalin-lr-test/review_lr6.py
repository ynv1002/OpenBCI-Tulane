from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.run_active_vs_rest_cross_session_calibrated import (
    DEFAULT_CALIBRATION_SEC,
    compute_calibration_stats,
    normalize_with_calibration,
)
from analysis.run_active_vs_rest_experiment import remap_active_vs_rest
from analysis.run_two_stage_lr_pipeline import ACTIVE_THRESHOLD, fit_active_gate, fit_left_right_model
from analysis.utils import (
    WINDOW_METADATA_COLUMNS,
    audit_all_sessions,
    audit_session,
    build_windows_for_audit,
    collapse_marker_events,
    compute_channel_quality,
    dataframe_to_markdown,
    ensure_output_dir,
    estimate_sampling,
    load_openbci_csv,
    pair_marker_events,
    preprocess_session_signals,
    select_model_channels,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
EXPECTED_MARKER_PATTERN = [1, 2] * 6 + [3, 4] * 6
COUNT_SMOOTHING_SAMPLES = 75
COUNT_MIN_PEAK_DISTANCE_SEC = 0.75
COUNT_PROMINENCE_STD_SCALE = 0.6
COUNT_MIN_PROMINENCE_UV = 0.02
COUNT_EARLY_BOUNDARY_EXCLUSION_SEC = 0.10
MODEL_WINDOW_SEC = 2.0
MODEL_OVERLAP = 0.5
MODEL_FEATURE_MODE = "combined"
MODEL_WITH_ASYMMETRY = True
MODEL_TARGET_SESSION_RANK = 999
MODEL_TARGET_DATE = "2026-04-08"
LEFT = "LEFT"
RIGHT = "RIGHT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-off trial and event review for the clean Dalin LR(6) recording."
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to Dalin-LR(6)-4:7.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for derived trial review outputs.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=250.0,
        help="Authoritative sample rate for reporting and event timing.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the debug trial review plot.",
    )
    return parser.parse_args()


def _fail(message: str) -> None:
    raise SystemExit(message)


def _smooth_values(values: np.ndarray, window_samples: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window_samples, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _build_trial_table(raw_df: pd.DataFrame, fs_hz: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if "Marker" not in raw_df.columns:
        _fail("The target LR dataset does not contain a Marker column.")

    marker_events = collapse_marker_events(raw_df["Marker"])
    marker_codes = [int(event["code"]) for event in marker_events]
    if len(marker_events) != len(EXPECTED_MARKER_PATTERN):
        _fail(
            f"Expected 24 collapsed markers, found {len(marker_events)}. "
            "This script only supports the clean 12-trial LR(6) structure."
        )
    if marker_codes != EXPECTED_MARKER_PATTERN:
        _fail(
            "Marker pattern did not match the trusted LR(6) layout. "
            f"Observed codes: {marker_codes}"
        )

    left_intervals, left_issues = pair_marker_events(marker_events, 1, 2)
    right_intervals, right_issues = pair_marker_events(marker_events, 3, 4)
    if left_issues or right_issues:
        _fail(
            "Marker pairing issues were detected in a file that should be clean: "
            + " | ".join(left_issues + right_issues)
        )
    if len(left_intervals) != 6 or len(right_intervals) != 6:
        _fail(
            "Expected exactly 6 left and 6 right trials. "
            f"Found {len(left_intervals)} left and {len(right_intervals)} right."
        )

    rows: list[dict[str, Any]] = []
    for index, interval in enumerate(left_intervals, start=1):
        start_sample = int(interval["start_sample"])
        end_sample = int(interval["end_sample"])
        rows.append(
            {
                "trial_id": index,
                "hand": LEFT,
                "expected_clench_count": index,
                "marker_pair_type": "1→2",
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time_sec": float(start_sample / fs_hz),
                "end_time_sec": float(end_sample / fs_hz),
                "duration_sec": float((end_sample - start_sample) / fs_hz),
            }
        )
    for index, interval in enumerate(right_intervals, start=1):
        start_sample = int(interval["start_sample"])
        end_sample = int(interval["end_sample"])
        rows.append(
            {
                "trial_id": 6 + index,
                "hand": RIGHT,
                "expected_clench_count": index,
                "marker_pair_type": "3→4",
                "start_sample": start_sample,
                "end_sample": end_sample,
                "start_time_sec": float(start_sample / fs_hz),
                "end_time_sec": float(end_sample / fs_hz),
                "duration_sec": float((end_sample - start_sample) / fs_hz),
            }
        )

    trials_df = pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)
    return trials_df, marker_events


def _target_count_channels(channel_quality_df: pd.DataFrame) -> list[str]:
    ordered = channel_quality_df.copy()
    ordered["channel_index"] = ordered["channel"].str.extract(r"(\d+)").astype(int)
    usable = ordered[ordered["status"] != "unsafe"].sort_values("channel_index")
    channels = usable["channel"].astype(str).tolist()
    if len(channels) < 2:
        _fail("Fewer than two target channels survived the unsafe-channel screen.")
    return channels


def _build_count_signal(raw_df: pd.DataFrame, channels: list[str], fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    filtered = preprocess_session_signals(raw_df, "left_right", channels, fs_hz)
    aggregate_rms = np.sqrt(np.mean(filtered.to_numpy(dtype=float) ** 2, axis=1))
    aggregate_rms_smooth = _smooth_values(aggregate_rms, COUNT_SMOOTHING_SAMPLES)
    return aggregate_rms, aggregate_rms_smooth


def _detect_trial_events(
    trials_df: pd.DataFrame,
    aggregate_rms: np.ndarray,
    fs_hz: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    min_distance_samples = max(1, int(round(COUNT_MIN_PEAK_DISTANCE_SEC * fs_hz)))

    for trial in trials_df.to_dict(orient="records"):
        start_sample = int(trial["start_sample"])
        end_sample = int(trial["end_sample"])
        trial_values = aggregate_rms[start_sample : end_sample + 1]
        trial_smooth = _smooth_values(trial_values, COUNT_SMOOTHING_SAMPLES)
        prominence_threshold = max(float(np.std(trial_smooth)) * COUNT_PROMINENCE_STD_SCALE, COUNT_MIN_PROMINENCE_UV)
        peak_indices, properties = find_peaks(
            trial_smooth,
            prominence=prominence_threshold,
            distance=min_distance_samples,
        )

        detected_times: list[float] = []
        dropped_reasons: list[str] = []
        kept_count = 0
        for event_index, peak_idx in enumerate(peak_indices, start=1):
            peak_sample = start_sample + int(peak_idx)
            peak_time_sec = float(peak_sample / fs_hz)
            peak_time_relative_sec = float(peak_time_sec - float(trial["start_time_sec"]))
            peak_prominence = float(properties["prominences"][event_index - 1])
            peak_value = float(trial_smooth[peak_idx])
            kept_for_count = peak_time_relative_sec >= COUNT_EARLY_BOUNDARY_EXCLUSION_SEC
            filter_reason = "" if kept_for_count else "early_boundary_exclusion"
            if kept_for_count:
                detected_times.append(round(peak_time_sec, 3))
                kept_count += 1
            else:
                dropped_reasons.append(
                    f"dropped onset-adjacent peak at {peak_time_relative_sec:.3f}s"
                )
            event_rows.append(
                {
                    "trial_id": int(trial["trial_id"]),
                    "hand": str(trial["hand"]),
                    "event_label": f"{trial['hand']}_CLICK",
                    "event_index": event_index,
                    "peak_sample": peak_sample,
                    "peak_time_sec": peak_time_sec,
                    "peak_time_relative_sec": peak_time_relative_sec,
                    "peak_value": peak_value,
                    "peak_prominence": peak_prominence,
                    "kept_for_count": bool(kept_for_count),
                    "filter_reason": filter_reason,
                }
            )

        estimated_count = int(kept_count)
        count_error = int(estimated_count - int(trial["expected_clench_count"]))
        count_match = bool(estimated_count == int(trial["expected_clench_count"]))
        if count_match:
            notes = "exact count match"
        elif abs(count_error) == 1:
            notes = "off by 1"
        else:
            notes = f"count error {count_error:+d}"
        if dropped_reasons:
            notes = notes + "; " + "; ".join(dropped_reasons)

        count_basis = (
            f"{estimated_count} peaks; smooth={COUNT_SMOOTHING_SAMPLES} samples; "
            f"min_distance={COUNT_MIN_PEAK_DISTANCE_SEC:.2f}s; "
            f"prominence>={prominence_threshold:.3f}uV; "
            f"ignore_peaks_before={COUNT_EARLY_BOUNDARY_EXCLUSION_SEC:.2f}s"
        )

        trial_rows.append(
            {
                **trial,
                "estimated_clench_count": estimated_count,
                "count_match": count_match,
                "count_error": count_error,
                "detected_event_times_sec": json.dumps(detected_times),
                "count_basis": count_basis,
                "notes": notes,
            }
        )

    return pd.DataFrame(trial_rows), pd.DataFrame(event_rows)


def _prepare_target_model_audit(csv_path: Path, fs_hz: float) -> dict[str, Any]:
    target_audit = audit_session(csv_path, "left_right")
    target_audit["session_rank"] = MODEL_TARGET_SESSION_RANK
    target_audit["parsed_date"] = MODEL_TARGET_DATE
    target_audit["sampling"] = {**target_audit["sampling"], "fs_used_hz": float(fs_hz)}
    return target_audit


def _normalize_training_windows(
    train_audits: list[dict[str, Any]],
    model_channels: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    train_frames: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None

    for audit in train_audits:
        frame = build_windows_for_audit(
            audit,
            model_channels,
            split_role="train",
            window_sec=MODEL_WINDOW_SEC,
            overlap=MODEL_OVERLAP,
            feature_mode=MODEL_FEATURE_MODE,
            with_asymmetry=MODEL_WITH_ASYMMETRY,
        )
        if feature_columns is None:
            feature_columns = [column for column in frame.columns if column not in WINDOW_METADATA_COLUMNS]
        mean, std, _ = compute_calibration_stats(frame, feature_columns, DEFAULT_CALIBRATION_SEC)
        train_frames.append(normalize_with_calibration(frame, feature_columns, mean, std))

    if not train_frames or feature_columns is None:
        _fail("Could not build training windows for the auxiliary LR comparison.")

    train_norm = pd.concat(train_frames, ignore_index=True).reset_index(drop=True)
    return train_norm, feature_columns


def _run_model_comparison(
    csv_path: Path,
    target_trial_channels: list[str],
    fs_hz: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    audits = audit_all_sessions()
    train_audits = [audit for audit in audits if audit["family"] == "left_right"][:-1]
    training_selected_channels, training_excluded_channels = select_model_channels(train_audits)
    model_channels = [channel for channel in training_selected_channels if channel in set(target_trial_channels)]
    if len(model_channels) < 2:
        return (
            pd.DataFrame(),
            {
                "training_selected_channels": training_selected_channels,
                "training_excluded_channels": training_excluded_channels,
                "model_channels": model_channels,
                "note": "Fewer than two overlapping channels survived for auxiliary LR comparison.",
            },
        )

    train_norm, feature_columns = _normalize_training_windows(train_audits, model_channels)

    target_audit = _prepare_target_model_audit(csv_path, fs_hz)
    target_frame = build_windows_for_audit(
        target_audit,
        model_channels,
        split_role="score",
        window_sec=MODEL_WINDOW_SEC,
        overlap=MODEL_OVERLAP,
        feature_mode=MODEL_FEATURE_MODE,
        with_asymmetry=MODEL_WITH_ASYMMETRY,
    )
    target_mean, target_std, target_calibration_windows = compute_calibration_stats(
        target_frame,
        feature_columns,
        DEFAULT_CALIBRATION_SEC,
    )
    target_norm = normalize_with_calibration(target_frame, feature_columns, target_mean, target_std).reset_index(drop=True)

    gate_train = remap_active_vs_rest(train_norm)
    _, p_active = fit_active_gate(gate_train, target_norm, feature_columns)
    lr_pred = fit_left_right_model(train_norm, target_norm, feature_columns)

    model_window_df = target_norm.loc[
        :,
        ["start_sample", "end_sample", "start_time_sec", "end_time_sec", "label", "segment_index"],
    ].copy()
    model_window_df["p_active"] = p_active.to_numpy(dtype=float)
    model_window_df["gate_active"] = (p_active >= ACTIVE_THRESHOLD).astype(int).to_numpy(dtype=int)
    model_window_df["lr_pred"] = lr_pred.astype(str).to_numpy(dtype=object)

    metadata = {
        "training_selected_channels": training_selected_channels,
        "training_excluded_channels": training_excluded_channels,
        "model_channels": model_channels,
        "train_files": [audit["filename"] for audit in train_audits],
        "target_calibration_windows": int(target_calibration_windows),
        "note": "Auxiliary LR comparison refit in memory; use as a transfer diagnostic only.",
    }
    return model_window_df, metadata


def _summarize_model_by_trial(
    trial_summary_df: pd.DataFrame,
    model_window_df: pd.DataFrame,
) -> pd.DataFrame:
    if model_window_df.empty:
        out = trial_summary_df.copy()
        out["model_mean_p_active"] = np.nan
        out["model_active_window_fraction"] = np.nan
        out["model_majority_hand_pred"] = "unknown"
        out["model_majority_hand_vote_fraction"] = np.nan
        out["notes"] = out["notes"].astype(str) + "; auxiliary LR comparison unavailable"
        return out

    rows: list[dict[str, Any]] = []
    for trial in trial_summary_df.to_dict(orient="records"):
        mask = (
            (model_window_df["start_time_sec"] >= float(trial["start_time_sec"]) - 1e-9)
            & (model_window_df["end_time_sec"] <= float(trial["end_time_sec"]) + 1e-9)
        )
        trial_windows = model_window_df.loc[mask].copy()
        majority_pred = "unknown"
        majority_vote_fraction = float("nan")
        notes = str(trial["notes"])
        if trial_windows.empty:
            model_mean_p_active = float("nan")
            model_active_window_fraction = float("nan")
            notes = notes + "; no auxiliary model windows landed inside trial"
        else:
            model_mean_p_active = float(trial_windows["p_active"].mean())
            model_active_window_fraction = float(trial_windows["gate_active"].mean())
            vote_counts = trial_windows["lr_pred"].astype(str).value_counts()
            if not vote_counts.empty:
                majority_pred = str(vote_counts.index[0])
                majority_vote_fraction = float(vote_counts.iloc[0] / len(trial_windows))
                if majority_pred != str(trial["hand"]):
                    notes = notes + "; auxiliary model hand mismatch"

        rows.append(
            {
                **trial,
                "model_mean_p_active": model_mean_p_active,
                "model_active_window_fraction": model_active_window_fraction,
                "model_majority_hand_pred": majority_pred,
                "model_majority_hand_vote_fraction": majority_vote_fraction,
                "notes": notes,
            }
        )

    ordered_columns = [
        "trial_id",
        "hand",
        "expected_clench_count",
        "estimated_clench_count",
        "count_match",
        "count_error",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "detected_event_times_sec",
        "count_basis",
        "model_mean_p_active",
        "model_active_window_fraction",
        "model_majority_hand_pred",
        "model_majority_hand_vote_fraction",
        "notes",
    ]
    return pd.DataFrame(rows)[ordered_columns].copy()


def _write_trial_plot(
    output_path: Path,
    raw_df: pd.DataFrame,
    aggregate_rms_smooth: np.ndarray,
    trials_df: pd.DataFrame,
    event_candidates_df: pd.DataFrame,
    fs_hz: float,
) -> str | None:
    mpl_config_dir = output_path.parent / ".mplconfig"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir.resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "matplotlib is unavailable; skipped trial_review.png"

    time_axis = np.arange(len(raw_df), dtype=float) / fs_hz
    fig, ax = plt.subplots(figsize=(14, 5))

    left_label_added = False
    right_label_added = False
    for trial in trials_df.to_dict(orient="records"):
        if trial["hand"] == LEFT:
            color = "#d9eaf7"
            label = "LEFT trial" if not left_label_added else None
            left_label_added = True
        else:
            color = "#f8dfd0"
            label = "RIGHT trial" if not right_label_added else None
            right_label_added = True
        ax.axvspan(
            float(trial["start_time_sec"]),
            float(trial["end_time_sec"]),
            color=color,
            alpha=0.35,
            label=label,
            zorder=0,
        )

    for marker_time in np.where(pd.to_numeric(raw_df["Marker"], errors="coerce").fillna(0.0).round().astype(int).to_numpy() != 0)[0]:
        ax.axvline(float(marker_time / fs_hz), color="0.80", linewidth=0.7, alpha=0.8, zorder=1)

    ax.plot(
        time_axis,
        aggregate_rms_smooth,
        color="tab:blue",
        linewidth=1.2,
        label="Aggregate RMS (smoothed)",
        zorder=2,
    )

    if not event_candidates_df.empty:
        kept_events = event_candidates_df[event_candidates_df["kept_for_count"] == True]
        left_events = kept_events[kept_events["hand"] == LEFT]
        right_events = kept_events[kept_events["hand"] == RIGHT]
        if not left_events.empty:
            ax.scatter(
                left_events["peak_time_sec"],
                left_events["peak_value"],
                color="tab:green",
                s=24,
                label="LEFT events",
                zorder=3,
            )
        if not right_events.empty:
            ax.scatter(
                right_events["peak_time_sec"],
                right_events["peak_value"],
                color="tab:red",
                s=24,
                label="RIGHT events",
                zorder=3,
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Aggregate RMS (uV)")
    ax.set_title("Dalin LR(6) trial review: trusted trial spans and detected events")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return None


def _count_interpretation(summary_df: pd.DataFrame) -> str:
    exact_matches = int(summary_df["count_match"].sum())
    off_by_one = int((summary_df["count_error"].abs() == 1).sum())
    remaining_trial_ids = summary_df.loc[~summary_df["count_match"], "trial_id"].astype(int).tolist()
    if exact_matches == 9 and remaining_trial_ids == [3, 4, 8]:
        return (
            "The conservative onset-boundary filter improves the event counter to 9/12 exact matches; "
            "trials 3, 4, and 8 remain under-counted and were left unresolved to avoid overfitting."
        )
    if exact_matches >= 8 and exact_matches + off_by_one == len(summary_df):
        return "Simple event counting tracks the known count ramp well enough to use this file as an event benchmark."
    if exact_matches >= 6:
        return "Simple event counting is directionally useful here, but some trials still need manual caution."
    return "Simple event counting is too unstable to treat this file as a reliable event benchmark."


def _model_interpretation(summary_df: pd.DataFrame) -> str:
    valid = summary_df[summary_df["model_majority_hand_pred"].astype(str) != "unknown"].copy()
    if valid.empty:
        return "The auxiliary LR model comparison could not be computed."

    exact = int((valid["model_majority_hand_pred"] == valid["hand"]).sum())
    right_trials = valid[valid["hand"] == RIGHT]
    if not right_trials.empty and (right_trials["model_majority_hand_pred"] == LEFT).all():
        return (
            "The existing cross-subject LR model did not transfer cleanly to Dalin; "
            "it collapsed to LEFT on every right-hand trial."
        )
    if exact < len(valid):
        return (
            "The existing cross-subject LR model showed weak transfer on Dalin and should stay diagnostic-only."
        )
    return "The auxiliary LR model was at least directionally consistent on this file."


def _future_detection_interpretation(summary_df: pd.DataFrame) -> str:
    count_exact = int(summary_df["count_match"].sum())
    valid = summary_df[summary_df["model_majority_hand_pred"].astype(str) != "unknown"].copy()
    model_exact = int((valid["model_majority_hand_pred"] == valid["hand"]).sum()) if not valid.empty else 0
    if count_exact >= 8 and (valid.empty or model_exact < len(valid)):
        return (
            "This dataset supports an event-first path: use trusted trial blocks to refine click counting, "
            "then retrain or adapt the LR hand classifier before relying on model-based hand labels live."
        )
    if count_exact >= 8:
        return "This dataset supports both event counting and hand discrimination as promising next steps."
    return "Use this dataset mainly to debug trial-local event features before pushing toward live click control."


def _format_summary_lines(
    csv_path: Path,
    output_dir: Path,
    trials_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    channel_quality_df: pd.DataFrame,
    count_channels: list[str],
    model_metadata: dict[str, Any],
    sampling_note: dict[str, Any],
    fs_hz: float,
    plot_warning: str | None,
) -> list[str]:
    left_exact = int(summary_df.loc[summary_df["hand"] == LEFT, "count_match"].sum())
    right_exact = int(summary_df.loc[summary_df["hand"] == RIGHT, "count_match"].sum())
    total_exact = int(summary_df["count_match"].sum())
    off_by_one = int(((summary_df["count_error"].abs() == 1) & (~summary_df["count_match"])).sum())
    remaining_trial_ids = summary_df.loc[~summary_df["count_match"], "trial_id"].astype(int).tolist()

    preview = summary_df.loc[
        :,
        [
            "trial_id",
            "hand",
            "expected_clench_count",
            "estimated_clench_count",
            "count_match",
            "model_majority_hand_pred",
            "model_mean_p_active",
        ],
    ].copy()
    preview["model_mean_p_active"] = preview["model_mean_p_active"].map(
        lambda value: "" if pd.isna(value) else f"{float(value):.3f}"
    )

    lines = [
        f"Dalin LR(6) review for {csv_path.name}",
        f"  Output directory: {output_dir}",
        f"  Fixed reporting rate: {fs_hz:.1f} Hz",
        f"  Timestamp-derived rate note: {sampling_note['fs_used_hz']:.3f} Hz ({' | '.join(sampling_note['notes'])})",
        f"  Trial count: {len(summary_df)}",
        f"  Counting channels: {', '.join(count_channels)}",
        f"  Auxiliary model channels: {', '.join(model_metadata.get('model_channels', [])) or 'none'}",
    ]

    questionable = channel_quality_df[channel_quality_df["status"] != "safe"].copy()
    if not questionable.empty:
        lines.append("  Channel warnings:")
        for _, row in questionable.iterrows():
            lines.append(f"    - {row['channel']}: {row['status']} ({row['reasons']})")

    lines.extend(
        [
            "",
            "Per-trial summary",
            dataframe_to_markdown(preview),
            "",
            "Count summary",
            f"  Exact matches: {total_exact}/12",
            f"  LEFT exact matches: {left_exact}/6",
            f"  RIGHT exact matches: {right_exact}/6",
            f"  Off-by-one trials: {off_by_one}",
            f"  Remaining unresolved trials: {remaining_trial_ids}",
            "",
            "Interpretation",
            f"  - {_count_interpretation(summary_df)}",
            f"  - {_model_interpretation(summary_df)}",
            f"  - {_future_detection_interpretation(summary_df)}",
        ]
    )
    if plot_warning:
        lines.append(f"  - Plot warning: {plot_warning}")
    return lines


def run_review(
    csv_path: Path,
    output_dir: Path,
    fs_hz: float,
    make_plot: bool,
) -> dict[str, Any]:
    if not csv_path.exists():
        _fail(f"CSV not found: {csv_path}")

    output_dir = ensure_output_dir(output_dir)
    raw_df = load_openbci_csv(csv_path.resolve())
    sampling_note = estimate_sampling(raw_df)
    trials_df, _ = _build_trial_table(raw_df, fs_hz)
    channel_quality_df = compute_channel_quality(raw_df)
    count_channels = _target_count_channels(channel_quality_df)
    aggregate_rms, aggregate_rms_smooth = _build_count_signal(raw_df, count_channels, fs_hz)
    counted_trials_df, event_candidates_df = _detect_trial_events(trials_df, aggregate_rms, fs_hz)
    model_window_df, model_metadata = _run_model_comparison(csv_path.resolve(), count_channels, fs_hz)
    trial_summary_df = _summarize_model_by_trial(counted_trials_df, model_window_df)

    trial_summary_path = output_dir / "trial_summary.csv"
    event_candidates_path = output_dir / "event_candidates.csv"
    trial_summary_df.to_csv(trial_summary_path, index=False)
    event_candidates_df.to_csv(event_candidates_path, index=False)

    plot_warning = None
    if make_plot:
        plot_warning = _write_trial_plot(
            output_dir / "trial_review.png",
            raw_df=raw_df,
            aggregate_rms_smooth=aggregate_rms_smooth,
            trials_df=trial_summary_df,
            event_candidates_df=event_candidates_df,
            fs_hz=fs_hz,
        )

    summary_lines = _format_summary_lines(
        csv_path=csv_path,
        output_dir=output_dir,
        trials_df=trials_df,
        summary_df=trial_summary_df,
        channel_quality_df=channel_quality_df,
        count_channels=count_channels,
        model_metadata=model_metadata,
        sampling_note=sampling_note,
        fs_hz=fs_hz,
        plot_warning=plot_warning,
    )
    print("\n".join(summary_lines))

    return {
        "trial_summary_path": trial_summary_path,
        "event_candidates_path": event_candidates_path,
        "trial_summary_df": trial_summary_df,
        "event_candidates_df": event_candidates_df,
        "count_channels": count_channels,
        "model_metadata": model_metadata,
        "plot_warning": plot_warning,
    }


def main() -> None:
    args = parse_args()
    run_review(
        csv_path=args.csv,
        output_dir=args.output_dir,
        fs_hz=float(args.fs),
        make_plot=not bool(args.no_plot),
    )


if __name__ == "__main__":
    main()
