from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .lrj_dataset import LRJSessionSpec, build_lrj_session_record, lrj_session_specs
from .utils import dataframe_to_markdown, ensure_output_dir, write_json


@dataclass(frozen=True)
class LRJReviewConfig:
    review_title: str
    console_title: str
    plot_title: str
    cli_description: str
    csv_help: str
    default_output_dir: Path


def build_arg_parser(config: LRJReviewConfig) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=config.cli_description)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help=config.csv_help,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.default_output_dir,
        help="Directory for derived LRJ review outputs.",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=250.0,
        help="Authoritative sample rate for reporting and trial timing.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the debug trial review plot.",
    )
    return parser


def _fail(message: str) -> None:
    raise SystemExit(message)


def _resolve_session_spec(config: LRJReviewConfig, csv_path: Path) -> LRJSessionSpec:
    resolved = csv_path.resolve()
    for session_spec in lrj_session_specs():
        if session_spec.csv_path.resolve() == resolved:
            return session_spec
    return LRJSessionSpec(
        subject="Custom",
        display_name=config.review_title,
        filename=resolved.name,
        csv_path=resolved,
    )


def _write_trial_plot(
    config: LRJReviewConfig,
    output_path: Path,
    raw_df: pd.DataFrame,
    trial_summary_df: pd.DataFrame,
    event_candidates_df: pd.DataFrame,
    count_signals: dict[str, dict[str, np.ndarray]],
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
    lr_signal = count_signals["left_right"]["aggregate_rms_smooth"]
    jaw_signal = count_signals["jaw"]["aggregate_rms_smooth"]

    fig, ax = plt.subplots(figsize=(14, 5))

    shade_colors = {
        "LEFT": "#d7efe4",
        "RIGHT": "#f7ddd6",
        "JAW": "#f7ead3",
    }
    peak_colors = {
        "LEFT": "tab:green",
        "RIGHT": "tab:red",
        "JAW": "tab:orange",
    }
    shown_labels: set[str] = set()
    for trial in trial_summary_df.to_dict(orient="records"):
        label = str(trial["label"])
        plot_label = f"{label} trial" if label not in shown_labels else None
        shown_labels.add(label)
        ax.axvspan(
            float(trial["start_time_sec"]),
            float(trial["end_time_sec"]),
            color=shade_colors.get(label, "#e5e5e5"),
            alpha=0.35,
            label=plot_label,
            zorder=0,
        )

    nonzero_markers = pd.to_numeric(raw_df["Marker"], errors="coerce").fillna(0.0).round().astype(int).to_numpy() != 0
    for marker_time in np.where(nonzero_markers)[0]:
        ax.axvline(float(marker_time / fs_hz), color="0.82", linewidth=0.7, alpha=0.8, zorder=1)

    ax.plot(
        time_axis,
        lr_signal,
        color="#2c7fb8",
        linewidth=1.1,
        alpha=0.85,
        label="LR envelope",
        zorder=2,
    )
    ax.plot(
        time_axis,
        jaw_signal,
        color="#d95f0e",
        linewidth=1.1,
        alpha=0.70,
        label="Jaw envelope",
        zorder=2,
    )

    kept_events = event_candidates_df[event_candidates_df["kept_for_count"] == True].copy()
    for label in ("LEFT", "RIGHT", "JAW"):
        label_events = kept_events[kept_events["label"] == label]
        if label_events.empty:
            continue
        ax.scatter(
            label_events["peak_time_sec"],
            label_events["peak_value"],
            color=peak_colors[label],
            s=24,
            label=f"{label} peaks",
            zorder=3,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Aggregate RMS (uV)")
    ax.set_title(f"{config.plot_title}: reconstructed trial spans and diagnostic count peaks")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return None


def _format_bullet_lines(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


def _build_summary_payload(
    config: LRJReviewConfig,
    csv_path: Path,
    output_dir: Path,
    fs_hz: float,
    sampling_note: dict[str, Any],
    count_channels: list[str],
    marker_audit_df: pd.DataFrame,
    trial_summary_df: pd.DataFrame,
    channel_quality_df: pd.DataFrame,
    hard_issues: list[str],
    soft_warnings: list[str],
    plot_warning: str | None,
) -> dict[str, Any]:
    count_mismatches = trial_summary_df[trial_summary_df["count_match"] == False].copy()
    questionable_channels = channel_quality_df[channel_quality_df["status"] != "safe"].copy()

    return {
        "review_title": config.review_title,
        "csv_path": str(csv_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "fs_hz": float(fs_hz),
        "sampling": sampling_note,
        "count_channels": count_channels,
        "structural_pass": not hard_issues,
        "marker_event_count": int(len(marker_audit_df)),
        "trial_count": int(len(trial_summary_df)),
        "trial_counts_by_label": {
            str(label): int(count)
            for label, count in trial_summary_df["label"].value_counts().sort_index().to_dict().items()
        },
        "count_matches": int(trial_summary_df["count_match"].sum()),
        "count_mismatches": int((trial_summary_df["count_match"] == False).sum()),
        "hard_issues": hard_issues,
        "soft_warnings": soft_warnings + ([plot_warning] if plot_warning else []),
        "questionable_channels": questionable_channels.to_dict(orient="records"),
        "count_mismatch_rows": count_mismatches[
            ["trial_index_overall", "label", "trial_index_within_label", "expected_count", "observed_count", "count_error"]
        ].to_dict(orient="records"),
        "paths": {
            "marker_audit_csv": str((output_dir / "marker_audit.csv").resolve()),
            "trial_summary_csv": str((output_dir / "trial_summary.csv").resolve()),
            "channel_quality_csv": str((output_dir / "channel_quality.csv").resolve()),
            "summary_json": str((output_dir / "summary.json").resolve()),
            "summary_md": str((output_dir / "summary.md").resolve()),
            "trial_review_png": str((output_dir / "trial_review.png").resolve()),
        },
    }


def _write_summary_markdown(
    output_path: Path,
    payload: dict[str, Any],
    marker_audit_df: pd.DataFrame,
    trial_summary_df: pd.DataFrame,
    channel_quality_df: pd.DataFrame,
) -> None:
    count_mismatches = trial_summary_df[trial_summary_df["count_match"] == False].copy()
    preview_columns = [
        "trial_index_overall",
        "label",
        "trial_index_within_label",
        "expected_count",
        "observed_count",
        "count_match",
        "duration_sec",
        "gap_from_previous_trial_sec",
        "issues",
    ]
    marker_preview = marker_audit_df[
        [
            "event_id",
            "marker_code",
            "label",
            "time_sec",
            "gap_from_previous_sec",
            "pair_index",
            "pair_role",
        ]
    ].copy()

    lines = [
        f"# {payload['review_title']}",
        "",
        f"- CSV: `{payload['csv_path']}`",
        f"- Output directory: `{payload['output_dir']}`",
        f"- Fixed reporting rate: `{payload['fs_hz']:.1f} Hz`",
        f"- Timestamp-derived rate used by audit helper: `{payload['sampling']['fs_used_hz']:.3f} Hz`",
        f"- Structural pass: `{payload['structural_pass']}`",
        f"- Marker events: `{payload['marker_event_count']}`",
        f"- Reconstructed trials: `{payload['trial_count']}`",
        f"- Trial counts by label: `{payload['trial_counts_by_label']}`",
        f"- Count matches: `{payload['count_matches']}/{payload['trial_count']}`",
        f"- Count channels: `{', '.join(payload['count_channels'])}`",
        "",
        "## Hard Issues",
        "",
        *_format_bullet_lines(payload["hard_issues"]),
        "",
        "## Soft Warnings",
        "",
        *_format_bullet_lines(payload["soft_warnings"]),
        "",
        "## Count Mismatches",
        "",
    ]

    if count_mismatches.empty:
        lines.append("No count mismatches were detected by the diagnostic peak counter.")
    else:
        lines.append(
            dataframe_to_markdown(
                count_mismatches[
                    [
                        "trial_index_overall",
                        "label",
                        "trial_index_within_label",
                        "expected_count",
                        "observed_count",
                        "count_error",
                        "issues",
                    ]
                ],
                include_index=False,
            )
        )

    lines.extend(
        [
            "",
            "## Trial Summary",
            "",
            dataframe_to_markdown(trial_summary_df[preview_columns], include_index=False),
            "",
            "## Channel Quality",
            "",
            dataframe_to_markdown(
                channel_quality_df[["channel", "status", "rail_fraction", "centered_std", "reasons"]],
                include_index=False,
            ),
            "",
            "## Marker Audit Preview",
            "",
            dataframe_to_markdown(marker_preview, include_index=False),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_lrj_review(
    config: LRJReviewConfig,
    csv_path: Path,
    output_dir: Path,
    fs_hz: float,
    make_plot: bool,
) -> dict[str, Any]:
    if not csv_path.exists():
        _fail(f"CSV not found: {csv_path}")

    output_dir = ensure_output_dir(output_dir)
    session_spec = _resolve_session_spec(config, csv_path)
    try:
        session_record = build_lrj_session_record(session_spec=session_spec, fs_hz=fs_hz)
    except ValueError as exc:
        _fail(str(exc))

    marker_audit_df = session_record["marker_audit_df"]
    trial_summary_df = session_record["trial_summary_df"]
    channel_quality_df = session_record["channel_quality_df"]
    count_channels = session_record["count_channels"]
    raw_df = session_record["raw_df"]
    event_candidates_df = session_record["event_candidates_df"]
    count_signals = session_record["count_signals"]
    sampling_note = session_record["sampling"]
    hard_issues = session_record["hard_issues"]
    soft_warnings = session_record["soft_warnings"]

    marker_audit_path = output_dir / "marker_audit.csv"
    trial_summary_path = output_dir / "trial_summary.csv"
    channel_quality_path = output_dir / "channel_quality.csv"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    marker_audit_df.to_csv(marker_audit_path, index=False)
    trial_summary_df.to_csv(trial_summary_path, index=False)
    channel_quality_df.to_csv(channel_quality_path, index=False)

    plot_warning = None
    if make_plot:
        plot_warning = _write_trial_plot(
            config,
            output_dir / "trial_review.png",
            raw_df=raw_df,
            trial_summary_df=trial_summary_df,
            event_candidates_df=event_candidates_df,
            count_signals=count_signals,
            fs_hz=fs_hz,
        )

    payload = _build_summary_payload(
        config=config,
        csv_path=csv_path,
        output_dir=output_dir,
        fs_hz=fs_hz,
        sampling_note=sampling_note,
        count_channels=count_channels,
        marker_audit_df=marker_audit_df,
        trial_summary_df=trial_summary_df,
        channel_quality_df=channel_quality_df,
        hard_issues=hard_issues,
        soft_warnings=soft_warnings,
        plot_warning=plot_warning,
    )
    write_json(summary_json_path, payload)
    _write_summary_markdown(summary_md_path, payload, marker_audit_df, trial_summary_df, channel_quality_df)

    pd.set_option("display.max_columns", 20)
    print(f"{config.console_title} for {csv_path.name}")
    print(f"  Output directory: {output_dir}")
    print(f"  Structural pass: {payload['structural_pass']}")
    print(f"  Marker events: {payload['marker_event_count']}")
    print(f"  Reconstructed trials: {payload['trial_count']}")
    print(f"  Count matches: {payload['count_matches']}/{payload['trial_count']}")
    print(f"  Count channels: {', '.join(count_channels)}")
    if hard_issues:
        print("  Hard issues:")
        for issue in hard_issues:
            print(f"    - {issue}")
    if payload["soft_warnings"]:
        print("  Soft warnings:")
        for warning in payload["soft_warnings"]:
            print(f"    - {warning}")
    print("\nTrial preview")
    print(
        trial_summary_df[
            [
                "trial_index_overall",
                "label",
                "trial_index_within_label",
                "expected_count",
                "observed_count",
                "count_match",
                "duration_sec",
            ]
        ].to_string(index=False)
    )

    return {
        "marker_audit_path": marker_audit_path,
        "trial_summary_path": trial_summary_path,
        "channel_quality_path": channel_quality_path,
        "summary_json_path": summary_json_path,
        "summary_md_path": summary_md_path,
        "plot_warning": plot_warning,
        "payload": payload,
    }
