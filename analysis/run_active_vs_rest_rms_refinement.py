from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
import sys
from typing import Dict, List, Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.run_active_vs_rest_experiment import (
        ACTIVE_LABEL,
        REST_LABEL,
        build_bundle_for_subset,
        class_counts,
        fit_and_score,
        resolve_left_right_subset,
    )
    from analysis.utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        LABEL_REST,
        audit_all_sessions,
        dataframe_to_markdown,
        ensure_output_dir,
        write_markdown,
    )
else:
    from .run_active_vs_rest_experiment import (
        ACTIVE_LABEL,
        REST_LABEL,
        build_bundle_for_subset,
        class_counts,
        fit_and_score,
        resolve_left_right_subset,
    )
    from .utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        LABEL_REST,
        audit_all_sessions,
        dataframe_to_markdown,
        ensure_output_dir,
        write_markdown,
    )


DEFAULT_TRAIN_SESSION = "LR-2-27-26-(01).csv"
DEFAULT_TEST_SESSION = "LR-3-15-26-(04).csv"
ACTIVE_SOURCE_LABELS = {"LEFT", "RIGHT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine ACTIVE labels by segment-local RMS and rerun ACTIVE vs REST."
    )
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--keep-fraction",
        action="append",
        type=float,
        default=None,
        help="Fraction of ACTIVE windows to keep within each ACTIVE segment. Repeat for multiple values. Defaults to 0.3, 0.4, 0.5.",
    )
    parser.add_argument(
        "--weak-active-policy",
        choices=["relabel_to_rest", "drop"],
        default="relabel_to_rest",
        help="How to handle weaker ACTIVE windows inside ACTIVE segments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for RMS-refined ACTIVE vs REST outputs.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def remap_active_vs_rest_with_rms_refinement(
    frame: pd.DataFrame,
    selected_channels: Sequence[str],
    keep_fraction: float,
    weak_active_policy: str,
) -> tuple[pd.DataFrame, Dict[str, int]]:
    out = frame.copy()
    out = out[out["label"].isin([LABEL_REST, *ACTIVE_SOURCE_LABELS])].copy()
    rms_columns = [f"{channel}_rms" for channel in selected_channels]
    missing_rms = [column for column in rms_columns if column not in out.columns]
    if missing_rms:
        raise ValueError(f"Missing RMS feature columns required for refinement: {missing_rms}")

    out["aggregate_rms"] = out.loc[:, rms_columns].mean(axis=1)
    out["original_label"] = out["label"]
    out["label"] = REST_LABEL

    kept_active_windows = 0
    relabeled_or_dropped_active_windows = 0
    grouped = out[out["original_label"].isin(ACTIVE_SOURCE_LABELS)].groupby(
        ["filename", "segment_index"], sort=False
    )
    drop_indices: List[int] = []

    for _, segment_df in grouped:
        keep_n = max(1, int(math.ceil(len(segment_df) * keep_fraction)))
        ranked = segment_df.sort_values(
            ["aggregate_rms", "start_time_sec"],
            ascending=[False, True],
        )
        keep_indices = ranked.index[:keep_n]
        kept_active_windows += int(len(keep_indices))
        weaker_indices = ranked.index[keep_n:]
        relabeled_or_dropped_active_windows += int(len(weaker_indices))

        out.loc[keep_indices, "label"] = ACTIVE_LABEL
        if weak_active_policy == "drop":
            drop_indices.extend(int(index) for index in weaker_indices.tolist())

    if weak_active_policy == "drop" and drop_indices:
        out = out.drop(index=drop_indices)

    stats = {
        "kept_active_windows": kept_active_windows,
        "weaker_active_windows": relabeled_or_dropped_active_windows,
    }
    return out.reset_index(drop=True), stats


def summarize_threshold_results(results_df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    per_threshold_best = (
        results_df.sort_values(["keep_fraction", "macro_f1", "accuracy"], ascending=[True, False, False])
        .groupby("keep_fraction", as_index=False)
        .first()
        .sort_values("keep_fraction")
        .reset_index(drop=True)
    )
    baseline_row = per_threshold_best.loc[per_threshold_best["keep_fraction"] == 1.0].iloc[0]
    non_baseline = per_threshold_best.loc[per_threshold_best["keep_fraction"] < 1.0].copy()
    best_refined = non_baseline.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0]

    lines = [
        f"- Baseline best (100% ACTIVE kept): `{baseline_row['model_name']}` with macro-F1 `{baseline_row['macro_f1']:.3f}` and ACTIVE recall `{baseline_row['active_recall']:.3f}`",
        f"- Best refined threshold: keep top `{int(round(best_refined['keep_fraction'] * 100))}%` ACTIVE windows with `{best_refined['model_name']}`",
        f"- Best refined macro-F1: `{best_refined['macro_f1']:.3f}`",
        f"- Best refined ACTIVE recall: `{best_refined['active_recall']:.3f}`",
    ]

    if float(best_refined["active_recall"]) > float(baseline_row["active_recall"]):
        lines.append("- ACTIVE recall improved relative to the unrefined baseline.")
    elif float(best_refined["active_recall"]) < float(baseline_row["active_recall"]):
        lines.append("- ACTIVE recall got worse relative to the unrefined baseline.")
    else:
        lines.append("- ACTIVE recall was unchanged relative to the unrefined baseline.")

    if float(best_refined["macro_f1"]) > float(baseline_row["macro_f1"]):
        lines.append("- Macro-F1 improved relative to the unrefined baseline.")
    elif float(best_refined["macro_f1"]) < float(baseline_row["macro_f1"]):
        lines.append("- Macro-F1 got worse relative to the unrefined baseline.")
    else:
        lines.append("- Macro-F1 was unchanged relative to the unrefined baseline.")

    return per_threshold_best, lines


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    keep_fractions = sorted(set(args.keep_fraction or [0.3, 0.4, 0.5]))
    keep_fractions = [1.0] + [fraction for fraction in keep_fractions if fraction < 1.0]

    audits = audit_all_sessions()
    subset_audits = resolve_left_right_subset(
        audits,
        [DEFAULT_TRAIN_SESSION, DEFAULT_TEST_SESSION],
    )

    rows: List[Dict[str, object]] = []
    for keep_fraction in keep_fractions:
        bundle = build_bundle_for_subset(
            subset_audits,
            window_sec=args.window_sec,
            overlap=args.overlap,
            with_asymmetry=True,
        )
        train_df, train_stats = remap_active_vs_rest_with_rms_refinement(
            bundle["train_windows"],
            selected_channels=bundle["selected_channels"],
            keep_fraction=keep_fraction,
            weak_active_policy=args.weak_active_policy,
        )
        test_df, test_stats = remap_active_vs_rest_with_rms_refinement(
            bundle["test_windows"],
            selected_channels=bundle["selected_channels"],
            keep_fraction=keep_fraction,
            weak_active_policy=args.weak_active_policy,
        )
        results = fit_and_score(train_df, test_df, bundle["feature_columns"])

        for model_name, metrics in results.items():
            threshold_tag = f"keep_{int(round(keep_fraction * 100)):03d}"
            confusion_csv = output_dir / (
                f"confusion_active_vs_rest_rms_refinement_{threshold_tag}_{args.weak_active_policy}_{model_name}.csv"
            )
            metrics["confusion_matrix"].to_csv(confusion_csv)
            row = {
                "family": "left_right",
                "task_type": "active_vs_rest_rms_refinement",
                "train_files": ", ".join(audit["filename"] for audit in bundle["train_audits"]),
                "test_file": bundle["test_audit"]["filename"],
                "feature_mode": "combined",
                "with_asymmetry": True,
                "weak_active_policy": args.weak_active_policy,
                "keep_fraction": keep_fraction,
                "keep_percent": int(round(keep_fraction * 100)),
                "model_name": model_name,
                "selected_channels": ", ".join(bundle["selected_channels"]),
                "feature_count": len(bundle["feature_columns"]),
                "train_class_counts": class_counts(train_df),
                "test_class_counts": class_counts(test_df),
                "train_kept_active_windows": train_stats["kept_active_windows"],
                "train_weaker_active_windows": train_stats["weaker_active_windows"],
                "test_kept_active_windows": test_stats["kept_active_windows"],
                "test_weaker_active_windows": test_stats["weaker_active_windows"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "active_precision": metrics["active_precision"],
                "active_recall": metrics["active_recall"],
                "active_to_rest_errors": int(metrics["confusion_matrix"].loc["true_ACTIVE", "pred_REST"]),
                "rest_to_active_errors": int(metrics["confusion_matrix"].loc["true_REST", "pred_ACTIVE"]),
                "confusion_matrix_csv": str(confusion_csv),
            }
            rows.append(row)

            print(
                f"[keep={row['keep_percent']}% | {model_name}] "
                f"accuracy={row['accuracy']:.3f} "
                f"macro_f1={row['macro_f1']:.3f} "
                f"active_precision={row['active_precision']:.3f} "
                f"active_recall={row['active_recall']:.3f}"
            )

    results_df = pd.DataFrame(rows).sort_values(
        ["keep_fraction", "macro_f1", "accuracy"], ascending=[True, False, False]
    ).reset_index(drop=True)
    results_csv = output_dir / f"active_vs_rest_rms_refinement_{slugify(args.weak_active_policy)}.csv"
    results_df.to_csv(results_csv, index=False)

    per_threshold_best, interpretation_lines = summarize_threshold_results(results_df)
    summary_df = results_df.loc[
        :,
        [
            "keep_percent",
            "model_name",
            "accuracy",
            "macro_f1",
            "active_precision",
            "active_recall",
            "train_kept_active_windows",
            "train_weaker_active_windows",
            "test_kept_active_windows",
            "test_weaker_active_windows",
            "active_to_rest_errors",
            "rest_to_active_errors",
        ],
    ].copy()
    best_overall = results_df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0]

    report_lines = [
        "# ACTIVE vs REST RMS Refinement Experiment",
        "",
        f"- Train file: `{DEFAULT_TRAIN_SESSION}`",
        f"- Test file: `{DEFAULT_TEST_SESSION}`",
        f"- Weak ACTIVE policy: `{args.weak_active_policy}`",
        f"- Output CSV: `{results_csv}`",
        f"- Best overall setup: keep top `{int(round(best_overall['keep_fraction'] * 100))}%` ACTIVE windows with `{best_overall['model_name']}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Best Per Threshold",
        "",
        dataframe_to_markdown(
            per_threshold_best.loc[
                :,
                [
                    "keep_percent",
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
        "## Interpretation",
        "",
        *interpretation_lines,
    ]
    summary_md = output_dir / f"active_vs_rest_rms_refinement_{slugify(args.weak_active_policy)}.md"
    write_markdown(summary_md, "\n".join(report_lines))

    print("\nBest overall setup:")
    print(
        f"- keep top {int(round(best_overall['keep_fraction'] * 100))}% ACTIVE windows with {best_overall['model_name']}"
    )
    print(f"- macro-F1={best_overall['macro_f1']:.3f}")
    print(f"- ACTIVE recall={best_overall['active_recall']:.3f}")
    print("\nInterpretation")
    for line in interpretation_lines:
        print(line)
    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
