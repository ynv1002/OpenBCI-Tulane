from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Dict, List, Sequence

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        LABEL_REST,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )
else:
    from .utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        LABEL_REST,
        WINDOW_METADATA_COLUMNS,
        audit_all_sessions,
        build_windows_for_audit,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        write_markdown,
    )


ACTIVE_LABEL = "ACTIVE"
REST_LABEL = "REST"
LABELS = [REST_LABEL, ACTIVE_LABEL]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACTIVE vs REST experiment on left/right EEG sessions.")
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument(
        "--include-session",
        action="append",
        default=None,
        help="Restrict the experiment to the listed left/right session filename(s). Use chronological order; the last included session is used as test.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional suffix for result filenames when running a custom subset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for ACTIVE vs REST outputs.",
    )
    return parser.parse_args()


def model_bank() -> Dict[str, object]:
    return {
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }


def remap_active_vs_rest(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out[out["label"].isin([LABEL_REST, "LEFT", "RIGHT"])].copy()
    out["label"] = out["label"].map(
        {
            LABEL_REST: REST_LABEL,
            "LEFT": ACTIVE_LABEL,
            "RIGHT": ACTIVE_LABEL,
        }
    )
    return out.reset_index(drop=True)


def class_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts = frame["label"].value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def resolve_left_right_subset(audits: Sequence[Dict[str, object]], include_sessions: Sequence[str] | None) -> List[Dict[str, object]]:
    family_audits = [audit for audit in audits if audit["family"] == "left_right"]
    if not include_sessions:
        return family_audits

    requested = set(include_sessions)
    subset = [audit for audit in family_audits if audit["filename"] in requested]
    missing = sorted(requested - {audit["filename"] for audit in subset})
    if missing:
        available = [audit["filename"] for audit in family_audits]
        raise ValueError(f"Could not find session(s) {missing}. Available left/right sessions: {available}")
    if len(subset) < 2:
        raise ValueError("Need at least two sessions to create a train/test split.")
    return subset


def build_bundle_for_subset(
    subset_audits: Sequence[Dict[str, object]],
    window_sec: float,
    overlap: float,
    with_asymmetry: bool,
) -> Dict[str, object]:
    if len(subset_audits) < 2:
        raise ValueError("Need at least two sessions in the subset.")

    train_audits = list(subset_audits[:-1])
    test_audit = subset_audits[-1]
    selected_channels, excluded_channels = select_model_channels(train_audits)
    if not selected_channels:
        raise ValueError("No modeling channels survived the training-session quality screen for the requested subset.")

    train_frames = [
        build_windows_for_audit(
            audit,
            selected_channels,
            split_role="train",
            window_sec=window_sec,
            overlap=overlap,
            feature_mode="combined",
            with_asymmetry=with_asymmetry,
        )
        for audit in train_audits
    ]
    test_windows = build_windows_for_audit(
        test_audit,
        selected_channels,
        split_role="test",
        window_sec=window_sec,
        overlap=overlap,
        feature_mode="combined",
        with_asymmetry=with_asymmetry,
    )
    train_windows = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    feature_columns = [column for column in train_windows.columns if column not in WINDOW_METADATA_COLUMNS]

    return {
        "train_audits": train_audits,
        "test_audit": test_audit,
        "selected_channels": selected_channels,
        "excluded_channels": excluded_channels,
        "train_windows": train_windows,
        "test_windows": test_windows,
        "feature_columns": feature_columns,
    }


def fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Dict[str, Dict[str, object]]:
    X_train = train_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    X_test = test_df.loc[:, list(feature_columns)].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    results: Dict[str, Dict[str, object]] = {}
    for model_name, model in model_bank().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=LABELS),
            index=[f"true_{label}" for label in LABELS],
            columns=[f"pred_{label}" for label in LABELS],
        )
        results[model_name] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "active_precision": float(
                precision_score(y_test, y_pred, pos_label=ACTIVE_LABEL, zero_division=0)
            ),
            "active_recall": float(
                recall_score(y_test, y_pred, pos_label=ACTIVE_LABEL, zero_division=0)
            ),
            "confusion_matrix": cm,
        }
    return results


def summarize_interpretation(results_df: pd.DataFrame) -> List[str]:
    best_row = results_df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0]
    best_macro_f1 = float(best_row["macro_f1"])
    no_asym_best = results_df.loc[~results_df["with_asymmetry"], "macro_f1"].max()
    with_asym_best = results_df.loc[results_df["with_asymmetry"], "macro_f1"].max()
    missed_intent = int(best_row["active_to_rest_errors"])
    false_triggers = int(best_row["rest_to_active_errors"])

    lines = [
        f"- Best setup: `{best_row['model_name']}` with `with_asymmetry={bool(best_row['with_asymmetry'])}`",
        f"- Best accuracy: `{best_row['accuracy']:.3f}`",
        f"- Best macro-F1: `{best_macro_f1:.3f}`",
    ]
    if best_macro_f1 > 0.80:
        lines.append("- ACTIVE vs REST is strong enough to look viable for control under the current threshold (> 0.80 macro-F1).")
    else:
        lines.append("- ACTIVE vs REST is still below the strong-control target of 0.80 macro-F1.")

    if with_asym_best > no_asym_best:
        lines.append("- Asymmetry helped the best observed result.")
    elif with_asym_best < no_asym_best:
        lines.append("- Asymmetry hurt the best observed result.")
    else:
        lines.append("- Asymmetry did not change the best observed result.")

    if missed_intent > false_triggers:
        lines.append("- Errors are dominated by ACTIVE -> REST misses, so the detector is more conservative than trigger-happy.")
    elif missed_intent < false_triggers:
        lines.append("- Errors are dominated by REST -> ACTIVE false triggers, which is riskier for control.")
    else:
        lines.append("- ACTIVE -> REST misses and REST -> ACTIVE false triggers are balanced in the best run.")
    return lines


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()
    subset_audits = resolve_left_right_subset(audits, args.include_session)
    dataset_condition = "all_sessions"
    if args.include_session:
        dataset_condition = f"custom_subset_{len(subset_audits)}_sessions"
    output_tag = args.tag or dataset_condition
    rows: List[Dict[str, object]] = []

    for with_asymmetry in (False, True):
        bundle = build_bundle_for_subset(
            subset_audits,
            window_sec=args.window_sec,
            overlap=args.overlap,
            with_asymmetry=with_asymmetry,
        )
        train_df = remap_active_vs_rest(bundle["train_windows"])
        test_df = remap_active_vs_rest(bundle["test_windows"])
        results = fit_and_score(train_df, test_df, bundle["feature_columns"])

        for model_name, metrics in results.items():
            confusion_name = (
                f"confusion_active_vs_rest_{slugify(output_tag)}_"
                f"{'with' if with_asymmetry else 'no'}_asym_{model_name}.csv"
            )
            confusion_csv = output_dir / confusion_name
            metrics["confusion_matrix"].to_csv(confusion_csv)
            rows.append(
                {
                    "family": "left_right",
                    "task_type": "active_vs_rest",
                    "dataset_condition": dataset_condition,
                    "feature_mode": "combined",
                    "with_asymmetry": with_asymmetry,
                    "model_name": model_name,
                    "train_files": ", ".join(audit["filename"] for audit in bundle["train_audits"]),
                    "test_file": bundle["test_audit"]["filename"],
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
                f"[with_asymmetry={with_asymmetry} | {model_name}] "
                f"accuracy={metrics['accuracy']:.3f} "
                f"macro_f1={metrics['macro_f1']:.3f} "
                f"active_precision={metrics['active_precision']:.3f} "
                f"active_recall={metrics['active_recall']:.3f}"
            )

    results_df = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=[False, False]).reset_index(drop=True)
    if output_tag == "all_sessions":
        results_csv = output_dir / "active_vs_rest_results.csv"
        summary_md = output_dir / "active_vs_rest_summary.md"
    else:
        results_csv = output_dir / f"active_vs_rest_results_{slugify(output_tag)}.csv"
        summary_md = output_dir / f"active_vs_rest_summary_{slugify(output_tag)}.md"
    results_df.to_csv(results_csv, index=False)

    best_row = results_df.iloc[0]
    summary_columns = [
        "with_asymmetry",
        "model_name",
        "feature_count",
        "accuracy",
        "macro_f1",
        "active_precision",
        "active_recall",
        "active_to_rest_errors",
        "rest_to_active_errors",
    ]
    summary_df = results_df.loc[:, summary_columns].copy()
    interpretation_lines = summarize_interpretation(results_df)

    report_lines = [
        "# ACTIVE vs REST Experiment",
        "",
        f"- Output CSV: `{results_csv}`",
        f"- Best model: `{best_row['model_name']}`",
        f"- Best asymmetry setting: `with_asymmetry={bool(best_row['with_asymmetry'])}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(summary_df),
        "",
        "## Interpretation",
        "",
        *interpretation_lines,
    ]
    write_markdown(summary_md, "\n".join(report_lines))

    print("Sessions used:", ", ".join(audit["filename"] for audit in subset_audits))
    print("\nBest model:", best_row["model_name"])
    print("Best asymmetry setting:", f"with_asymmetry={bool(best_row['with_asymmetry'])}")
    print("\nInterpretation")
    for line in interpretation_lines:
        print(line)
    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
