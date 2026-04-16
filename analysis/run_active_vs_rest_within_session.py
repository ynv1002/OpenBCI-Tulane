from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Dict, List, Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.run_active_vs_rest_experiment import (
        class_counts,
        fit_and_score,
        remap_active_vs_rest,
    )
    from analysis.utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )
else:
    from .run_active_vs_rest_experiment import (
        class_counts,
        fit_and_score,
        remap_active_vs_rest,
    )
    from .utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )


DEFAULT_FILENAME = "LR-2-27-26-(01).csv"
DEFAULT_SPLIT_FRACTION = 0.70
DEFAULT_CROSS_SESSION_RESULTS = (
    Path(__file__).resolve().parent / "outputs" / "active_vs_rest_results_feb27_mar15.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ACTIVE vs REST within a single left/right session using a chronological split."
    )
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_SPLIT_FRACTION)
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for within-session outputs.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def resolve_session(audits: Sequence[Dict[str, object]], filename: str) -> Dict[str, object]:
    for audit in audits:
        if audit["family"] == "left_right" and audit["filename"] == filename:
            return audit
    available = [audit["filename"] for audit in audits if audit["family"] == "left_right"]
    raise ValueError(f"Could not find session '{filename}'. Available left/right sessions: {available}")


def build_within_session_bundle(
    audit: Dict[str, object],
    window_sec: float,
    overlap: float,
) -> Dict[str, object]:
    selected_channels, excluded_channels = select_model_channels([audit])
    if not selected_channels:
        raise ValueError("No modeling channels survived the session quality screen.")

    windows = build_windows_for_audit(
        audit,
        selected_channels,
        split_role="within_session",
        window_sec=window_sec,
        overlap=overlap,
        feature_mode="combined",
        with_asymmetry=True,
    )
    feature_columns = [column for column in windows.columns if column not in WINDOW_METADATA_COLUMNS]
    return {
        "audit": audit,
        "selected_channels": selected_channels,
        "excluded_channels": excluded_channels,
        "windows": windows,
        "feature_columns": feature_columns,
    }


def chronological_split(frame: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["start_time_sec", "start_sample"]).reset_index(drop=True)
    split_index = max(1, min(len(ordered) - 1, int(round(len(ordered) * train_fraction))))
    train_df = ordered.iloc[:split_index].copy().reset_index(drop=True)
    test_df = ordered.iloc[split_index:].copy().reset_index(drop=True)
    return train_df, test_df


def load_cross_session_best(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    best = df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0]
    return {
        "model_name": best["model_name"],
        "accuracy": float(best["accuracy"]),
        "macro_f1": float(best["macro_f1"]),
        "active_precision": float(best["active_precision"]),
        "active_recall": float(best["active_recall"]),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")

    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()
    audit = resolve_session(audits, args.filename)
    bundle = build_within_session_bundle(audit, args.window_sec, args.overlap)
    frame = remap_active_vs_rest(bundle["windows"])
    train_df, test_df = chronological_split(frame, args.train_fraction)
    results = fit_and_score(train_df, test_df, bundle["feature_columns"])

    rows: List[Dict[str, object]] = []
    session_tag = slugify(args.filename)
    for model_name, metrics in results.items():
        confusion_csv = output_dir / f"confusion_active_vs_rest_within_session_{session_tag}_{model_name}.csv"
        metrics["confusion_matrix"].to_csv(confusion_csv)
        rows.append(
            {
                "family": "left_right",
                "task_type": "active_vs_rest_within_session",
                "filename": audit["filename"],
                "train_fraction": args.train_fraction,
                "feature_mode": "combined",
                "with_asymmetry": True,
                "model_name": model_name,
                "selected_channels": ", ".join(bundle["selected_channels"]),
                "excluded_channels": " | ".join(
                    f"{key}: {value}" for key, value in bundle["excluded_channels"].items()
                ),
                "feature_count": len(bundle["feature_columns"]),
                "train_class_counts": class_counts(train_df),
                "test_class_counts": class_counts(test_df),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "active_precision": metrics["active_precision"],
                "active_recall": metrics["active_recall"],
                "active_to_rest_errors": int(metrics["confusion_matrix"].loc["true_ACTIVE", "pred_REST"]),
                "rest_to_active_errors": int(metrics["confusion_matrix"].loc["true_REST", "pred_ACTIVE"]),
                "confusion_matrix_csv": str(confusion_csv),
            }
        )
        print(
            f"[{model_name}] accuracy={metrics['accuracy']:.3f} "
            f"macro_f1={metrics['macro_f1']:.3f} "
            f"active_precision={metrics['active_precision']:.3f} "
            f"active_recall={metrics['active_recall']:.3f}"
        )

    results_df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    results_csv = output_dir / f"active_vs_rest_within_session_{session_tag}.csv"
    results_df.to_csv(results_csv, index=False)

    best_row = results_df.iloc[0]
    cross_session_best = load_cross_session_best(DEFAULT_CROSS_SESSION_RESULTS)
    comparison_lines: List[str] = []
    if cross_session_best is None:
        comparison_lines.append("- Cross-session comparison file was not found, so no numeric comparison was added.")
    else:
        comparison_lines.extend(
            [
                f"- Best within-session setup: `{best_row['model_name']}` with macro-F1 `{best_row['macro_f1']:.3f}` and ACTIVE recall `{best_row['active_recall']:.3f}`",
                f"- Best cross-session setup (Feb 27 -> March 15): `{cross_session_best['model_name']}` with macro-F1 `{cross_session_best['macro_f1']:.3f}` and ACTIVE recall `{cross_session_best['active_recall']:.3f}`",
                f"- Macro-F1 difference (within - cross): `{float(best_row['macro_f1']) - cross_session_best['macro_f1']:+.3f}`",
                f"- ACTIVE recall difference (within - cross): `{float(best_row['active_recall']) - cross_session_best['active_recall']:+.3f}`",
            ]
        )

    report_lines = [
        "# ACTIVE vs REST Within-Session Experiment",
        "",
        f"- Session: `{audit['filename']}`",
        f"- Chronological split: first `{args.train_fraction:.0%}` train, last `{1.0 - args.train_fraction:.0%}` test",
        f"- Output CSV: `{results_csv}`",
        f"- Selected channels: `{', '.join(bundle['selected_channels'])}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(
            results_df.loc[
                :,
                [
                    "model_name",
                    "accuracy",
                    "macro_f1",
                    "active_precision",
                    "active_recall",
                    "active_to_rest_errors",
                    "rest_to_active_errors",
                ],
            ]
        ),
        "",
        "## Cross-Session Comparison",
        "",
        *comparison_lines,
    ]
    summary_md = output_dir / f"active_vs_rest_within_session_{session_tag}.md"
    write_markdown(summary_md, "\n".join(report_lines))

    print("\nBest within-session model:", best_row["model_name"])
    print(f"Best within-session macro-F1: {best_row['macro_f1']:.3f}")
    print(f"Best within-session ACTIVE recall: {best_row['active_recall']:.3f}")
    if cross_session_best is not None:
        print("\nCross-session comparison")
        for line in comparison_lines:
            print(line)
    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
