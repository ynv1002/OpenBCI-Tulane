from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from analysis.lr_event_validation.review_yaniv_lr import DEFAULT_INPUT_DIR as LR_INPUT_DIR
from analysis.lr_event_validation.review_yaniv_lr import run_validation
from analysis.stepwise_protocol_registry import (
    all_file_contracts,
    eeg_lr_main_contracts,
    lrj_main_contracts,
)
from analysis.utils import (
    compute_channel_quality,
    load_openbci_csv,
    preprocess_session_signals,
    write_json,
)

DEFAULT_VALIDATION_DIR = Path(__file__).resolve().parents[1] / "lr_event_validation" / "outputs"
DEFAULT_FS_HZ = 250.0
EVENT_WINDOW_SEC = 0.50
LEFT = "LEFT"
RIGHT = "RIGHT"
LABELS = [LEFT, RIGHT]
DEFAULT_INCLUDE_FILES = [contract.filename for contract in eeg_lr_main_contracts()]


def _fail(message: str) -> None:
    raise SystemExit(message)


def _target_count_channels(channel_quality_df: pd.DataFrame) -> list[str]:
    ordered = channel_quality_df.copy()
    ordered["channel_index"] = ordered["channel"].str.extract(r"(\d+)").astype(int)
    usable = ordered[ordered["status"] != "unsafe"].sort_values("channel_index")
    channels = usable["channel"].astype(str).tolist()
    if len(channels) < 2:
        _fail("Fewer than two target channels survived the unsafe-channel screen.")
    return channels


def resolve_include_files(include_files: list[str] | None) -> list[str]:
    if not include_files:
        include_files = list(DEFAULT_INCLUDE_FILES)
    deduped = list(dict.fromkeys(str(filename) for filename in include_files))
    if len(deduped) < 2:
        _fail("Need at least two files to run leave-one-session-out evaluation.")

    available_files = [contract.filename for contract in all_file_contracts()]
    missing = [filename for filename in deduped if filename not in available_files]
    if missing:
        _fail(f"Requested file(s) not found in registry: {missing}")
    return deduped


def _get_csv_path(filename: str) -> Path:
    for contract in all_file_contracts():
        if contract.filename == filename:
            return contract.csv_path
    _fail(f"Unknown file: {filename}")


def _ensure_lrj_validation_outputs(filename: str, validation_dir: Path, fs_hz: float) -> None:
    stem = Path(filename).stem
    file_dir = validation_dir / stem
    if (file_dir / "event_table.csv").exists() and (file_dir / "block_summary.csv").exists():
        return
        
    from analysis.lrj_dataset import lrj_session_specs, build_lrj_session_record
    file_dir.mkdir(parents=True, exist_ok=True)
    
    spec = next((s for s in lrj_session_specs() if s.filename == filename), None)
    if not spec:
        _fail(f"Could not find LRJSessionSpec for {filename}")
        
    record = build_lrj_session_record(spec, fs_hz)
    
    block_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    event_id_counter = 1
    
    for trial in record["trials_df"].to_dict(orient="records"):
        side = str(trial["label"])
        if side not in (LEFT, RIGHT):
            continue
            
        block_id = int(trial["trial_index_overall"])
        events = record["event_candidates_df"][
            (record["event_candidates_df"]["trial_index_overall"] == block_id) &
            (record["event_candidates_df"]["kept_for_count"] == True)
        ]
        
        detected_times = [float(row["peak_time_sec"]) for row in events.to_dict(orient="records")]
        
        block_rows.append({
            "file": filename,
            "block_id": block_id,
            "side": side,
            "start_time_sec": float(trial["start_time_sec"]),
            "end_time_sec": float(trial["end_time_sec"]),
            "duration_sec": float(trial["duration_sec"]),
            "detected_event_count": len(detected_times),
            "kept_event_times_sec": str(detected_times),
            "max_trace_value": 0.0,
            "mean_trace_value": 0.0,
            "notes": ""
        })
        
        for cand in events.to_dict(orient="records"):
            event_rows.append({
                "file": filename,
                "event_id": event_id_counter,
                "event_time_sec": float(cand["peak_time_sec"]),
                "event_time_relative_sec": float(cand["peak_time_relative_sec"]),
                "peak_value": float(cand["peak_value"]),
                "peak_prominence": float(cand["peak_prominence"]),
                "kept_for_count": True,
                "filter_reason": "",
                "assigned_block_id": block_id,
                "assigned_side": side,
                "inside_marker_block": "yes",
                "notes": ""
            })
            event_id_counter += 1
            
    pd.DataFrame(block_rows).to_csv(file_dir / "block_summary.csv", index=False)
    pd.DataFrame(event_rows).to_csv(file_dir / "event_table.csv", index=False)
    write_json(file_dir / "run_notes.json", {"source": "lrj_adapter"})


def _ensure_validation_outputs(validation_dir: Path, fs_hz: float, include_files: list[str]) -> None:
    missing_lr = []
    lrj_contracts = [c.filename for c in lrj_main_contracts()]
    for filename in include_files:
        stem = Path(filename).stem
        if not (validation_dir / stem / "event_table.csv").exists():
            if filename in lrj_contracts:
                _ensure_lrj_validation_outputs(filename, validation_dir, fs_hz)
            else:
                missing_lr.append(filename)
    if missing_lr:
        run_validation(
            input_dir=LR_INPUT_DIR,
            output_dir=validation_dir,
            fs_hz=fs_hz,
            make_plot=False,
        )


def _load_validation_bundle(validation_dir: Path, filename: str) -> dict[str, Any]:
    stem = Path(filename).stem
    file_dir = validation_dir / stem
    event_table_path = file_dir / "event_table.csv"
    block_summary_path = file_dir / "block_summary.csv"
    run_notes_path = file_dir / "run_notes.json"
    if not event_table_path.exists() or not block_summary_path.exists() or not run_notes_path.exists():
        _fail(f"Missing validation outputs for {filename} in {file_dir}")

    event_df = pd.read_csv(event_table_path)
    block_df = pd.read_csv(block_summary_path)
    run_notes = json.loads(run_notes_path.read_text(encoding="utf-8"))
    kept_df = event_df[
        (event_df["kept_for_count"] == True)
        & (event_df["inside_marker_block"] == "yes")
        & (event_df["assigned_side"].isin(LABELS))
    ].copy()
    kept_df["file"] = filename
    block_df["file"] = filename
    return {
        "filename": filename,
        "file_dir": file_dir,
        "event_df": kept_df.reset_index(drop=True),
        "block_df": block_df.reset_index(drop=True),
        "run_notes": run_notes,
    }


def _block_lookup(block_df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for row in block_df.to_dict(orient="records"):
        lookup[int(row["block_id"])] = row
    return lookup


def _compute_common_channels(file_paths: list[Path]) -> list[str]:
    channel_sets: list[set[str]] = []
    for path in file_paths:
        raw_df = load_openbci_csv(path)
        quality_df = compute_channel_quality(raw_df)
        channel_sets.append(set(_target_count_channels(quality_df)))
    common = sorted(set.intersection(*channel_sets), key=lambda name: int(name.split("_")[1]))
    if len(common) < 2:
        _fail("Fewer than two common non-unsafe channels survived across the high-trust sessions.")
    return common


def _window_bounds(
    peak_sample: int,
    block_start_sample: int,
    block_end_sample: int,
    window_samples: int,
) -> tuple[int, int]:
    half_left = window_samples // 2
    half_right = window_samples - half_left - 1
    start = peak_sample - half_left
    end = peak_sample + half_right

    if start < block_start_sample:
        shift = block_start_sample - start
        start += shift
        end += shift
    if end > block_end_sample:
        shift = end - block_end_sample
        start -= shift
        end -= shift

    start = max(block_start_sample, start)
    end = min(block_end_sample, end)

    current_len = end - start + 1
    if current_len < window_samples:
        deficit = window_samples - current_len
        grow_left = min(deficit // 2 + deficit % 2, start - block_start_sample)
        grow_right = min(deficit // 2, block_end_sample - end)
        start -= grow_left
        end += grow_right
        remaining = window_samples - (end - start + 1)
        if remaining > 0:
            extra_left = min(remaining, start - block_start_sample)
            start -= extra_left
            remaining -= extra_left
        if remaining > 0:
            extra_right = min(remaining, block_end_sample - end)
            end += extra_right

    return int(start), int(end)


def build_event_matrix(
    feature_extractor: Callable[[np.ndarray, list[str]], dict[str, float]],
    include_files: list[str] | None = None,
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
    fs_hz: float = DEFAULT_FS_HZ,
    window_sec: float = EVENT_WINDOW_SEC,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Builds the canonical training matrix across the requested files.
    Returns:
        event_matrix_df: DataFrame containing metadata and features for every valid event.
        feature_columns: The keys of the extracted features to use as X.
        common_channels: The active channel set computed.
    """
    resolved_files = resolve_include_files(include_files)
    validation_dir.mkdir(parents=True, exist_ok=True)
    _ensure_validation_outputs(validation_dir, fs_hz, resolved_files)

    file_paths = [_get_csv_path(filename) for filename in resolved_files]
    common_channels = _compute_common_channels(file_paths)
    validation_bundles = [_load_validation_bundle(validation_dir, filename) for filename in resolved_files]
    
    window_samples = max(3, int(round(window_sec * fs_hz)))
    rows: list[dict[str, Any]] = []
    
    for bundle in validation_bundles:
        filename = bundle["filename"]
        csv_path = _get_csv_path(filename)
        raw_df = load_openbci_csv(csv_path)
        processed = preprocess_session_signals(raw_df, "left_right", common_channels, fs_hz)
        block_lookup = _block_lookup(bundle["block_df"])

        for event in bundle["event_df"].to_dict(orient="records"):
            block_id = int(event["assigned_block_id"])
            block = block_lookup[block_id]
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
            
            features = feature_extractor(window, common_channels)

            row = {
                "file": filename,
                "event_id": int(event["event_id"]),
                "event_time_sec": float(event["event_time_sec"]),
                "event_time_relative_sec": float(event["event_time_relative_sec"]),
                "block_id": block_id,
                "label": str(event["assigned_side"]),
                "block_start_time_sec": float(block["start_time_sec"]),
                "block_end_time_sec": float(block["end_time_sec"]),
                "peak_value": float(event["peak_value"]),
                "peak_prominence": float(event["peak_prominence"]),
                "window_start_time_sec": float(window_start / fs_hz),
                "window_end_time_sec": float(window_end / fs_hz),
                "window_sec": float((window_end - window_start + 1) / fs_hz),
            }
            row.update(features)
            rows.append(row)

    event_matrix_df = pd.DataFrame(rows).sort_values(["file", "event_time_sec"]).reset_index(drop=True)
    metadata_columns = {
        "file",
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
    feature_columns = [column for column in event_matrix_df.columns if column not in metadata_columns]
    
    return event_matrix_df, feature_columns, common_channels


def build_tensor_matrix(
    include_files: list[str] | None = None,
    validation_dir: Path = DEFAULT_VALIDATION_DIR,
    fs_hz: float = DEFAULT_FS_HZ,
    window_sec: float = EVENT_WINDOW_SEC,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """
    Builds the canonical training matrix across the requested files, returning pure dimensional tensors.
    Returns:
        X_tensor: np.ndarray of shape (n_trials, n_channels, n_time_steps).
        metadata_df: DataFrame containing metadata (labels, file, event_id, etc.) for each trial.
        common_channels: The active channel set computed.
    """
    resolved_files = resolve_include_files(include_files)
    validation_dir.mkdir(parents=True, exist_ok=True)
    _ensure_validation_outputs(validation_dir, fs_hz, resolved_files)

    file_paths = [_get_csv_path(filename) for filename in resolved_files]
    common_channels = _compute_common_channels(file_paths)
    validation_bundles = [_load_validation_bundle(validation_dir, filename) for filename in resolved_files]
    
    window_samples = max(3, int(round(window_sec * fs_hz)))
    rows: list[dict[str, Any]] = []
    tensors: list[np.ndarray] = []
    
    for bundle in validation_bundles:
        filename = bundle["filename"]
        csv_path = _get_csv_path(filename)
        raw_df = load_openbci_csv(csv_path)
        processed = preprocess_session_signals(raw_df, "left_right", common_channels, fs_hz)
        block_lookup = _block_lookup(bundle["block_df"])

        for event in bundle["event_df"].to_dict(orient="records"):
            block_id = int(event["assigned_block_id"])
            block = block_lookup[block_id]
            peak_sample = int(round(float(event["event_time_sec"]) * fs_hz))
            block_start_sample = int(round(float(block["start_time_sec"]) * fs_hz))
            block_end_sample = int(round(float(block["end_time_sec"]) * fs_hz))
            window_start, window_end = _window_bounds(
                peak_sample=peak_sample,
                block_start_sample=block_start_sample,
                block_end_sample=block_end_sample,
                window_samples=window_samples,
            )
            
            # shape: (n_time_steps, n_channels)
            window = processed.iloc[window_start : window_end + 1].to_numpy(dtype=float)
            
            # expected CSP native shape: (n_channels, n_time_steps)
            tensors.append(window.T)

            row = {
                "file": filename,
                "event_id": int(event["event_id"]),
                "event_time_sec": float(event["event_time_sec"]),
                "event_time_relative_sec": float(event["event_time_relative_sec"]),
                "block_id": block_id,
                "label": str(event["assigned_side"]),
                "block_start_time_sec": float(block["start_time_sec"]),
                "block_end_time_sec": float(block["end_time_sec"]),
                "peak_value": float(event["peak_value"]),
                "peak_prominence": float(event["peak_prominence"]),
                "window_start_time_sec": float(window_start / fs_hz),
                "window_end_time_sec": float(window_end / fs_hz),
                "window_sec": float((window_end - window_start + 1) / fs_hz),
            }
            rows.append(row)

    metadata_df = pd.DataFrame(rows)
    # Ensure they are sorted consistently
    sort_idx = metadata_df.sort_values(["file", "event_time_sec"]).index
    metadata_df = metadata_df.iloc[sort_idx].reset_index(drop=True)
    X_tensor = np.array(tensors)[sort_idx]
    
    return X_tensor, metadata_df, common_channels
