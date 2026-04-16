from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.run_active_vs_rest_experiment import ACTIVE_LABEL, REST_LABEL, model_bank, remap_active_vs_rest
    from analysis.run_active_vs_rest_within_session import (
        DEFAULT_FILENAME,
        DEFAULT_SPLIT_FRACTION,
        build_within_session_bundle,
        chronological_split,
        resolve_session,
        slugify,
    )
    from analysis.utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown
else:
    from .run_active_vs_rest_experiment import ACTIVE_LABEL, REST_LABEL, model_bank, remap_active_vs_rest
    from .run_active_vs_rest_within_session import (
        DEFAULT_FILENAME,
        DEFAULT_SPLIT_FRACTION,
        build_within_session_bundle,
        chronological_split,
        resolve_session,
        slugify,
    )
    from .utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_markdown


DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep RandomForest ACTIVE probability thresholds for the within-session LR experiment."
    )
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_SPLIT_FRACTION)
    parser.add_argument(
        "--threshold",
        action="append",
        type=float,
        default=None,
        help="Decision threshold for P(ACTIVE). Repeat to sweep multiple values. Defaults to 0.3, 0.4, 0.5, 0.6.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for threshold sweep outputs.",
    )
    return parser.parse_args()


def evaluate_thresholds(
    y_true: pd.Series,
    active_prob: pd.Series,
    thresholds: List[float],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for threshold in thresholds:
        y_pred = pd.Series(
            [ACTIVE_LABEL if prob >= threshold else REST_LABEL for prob in active_prob],
            index=y_true.index,
        )
        cm = pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=[REST_LABEL, ACTIVE_LABEL]),
            index=[f"true_{REST_LABEL}", f"true_{ACTIVE_LABEL}"],
            columns=[f"pred_{REST_LABEL}", f"pred_{ACTIVE_LABEL}"],
        )
        rows.append(
            {
                "threshold": threshold,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "active_precision": float(
                    precision_score(y_true, y_pred, pos_label=ACTIVE_LABEL, zero_division=0)
                ),
                "active_recall": float(
                    recall_score(y_true, y_pred, pos_label=ACTIVE_LABEL, zero_division=0)
                ),
                "misses": int(cm.loc[f"true_{ACTIVE_LABEL}", f"pred_{REST_LABEL}"]),
                "false_triggers": int(cm.loc[f"true_{REST_LABEL}", f"pred_{ACTIVE_LABEL}"]),
                "confusion_matrix": cm,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")

    thresholds = sorted(set(args.threshold or DEFAULT_THRESHOLDS))
    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()
    audit = resolve_session(audits, args.filename)
    bundle = build_within_session_bundle(audit, window_sec=2.0, overlap=0.5)
    frame = remap_active_vs_rest(bundle["windows"])
    train_df, test_df = chronological_split(frame, args.train_fraction)

    model = model_bank()["RandomForest"]
    X_train = train_df.loc[:, bundle["feature_columns"]].to_numpy(dtype=float)
    X_test = test_df.loc[:, bundle["feature_columns"]].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    y_test = pd.Series(test_df["label"].to_numpy(), name="label")

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    active_index = list(model.classes_).index(ACTIVE_LABEL)
    active_prob = pd.Series(probs[:, active_index], index=test_df.index, name="p_active")

    sweep_df = evaluate_thresholds(y_test, active_prob, thresholds)
    session_tag = slugify(args.filename)
    results_csv = output_dir / "active_vs_rest_threshold_sweep.csv"
    sweep_df.loc[:, ["threshold", "accuracy", "macro_f1", "active_precision", "active_recall", "misses", "false_triggers"]].to_csv(
        results_csv,
        index=False,
    )

    for _, row in sweep_df.iterrows():
        confusion_csv = output_dir / (
            f"confusion_active_vs_rest_threshold_sweep_{session_tag}_{str(row['threshold']).replace('.', '_')}.csv"
        )
        row["confusion_matrix"].to_csv(confusion_csv)

    best_macro = sweep_df.sort_values(["macro_f1", "accuracy"], ascending=[False, False]).iloc[0]
    best_recall = sweep_df.sort_values(["active_recall", "macro_f1"], ascending=[False, False]).iloc[0]
    balance_df = sweep_df.copy()
    balance_df["balance_score"] = (
        balance_df["macro_f1"] - 0.001 * balance_df["false_triggers"] + 0.001 * balance_df["active_recall"] * 100.0
    )
    best_balance = balance_df.sort_values(["balance_score", "macro_f1"], ascending=[False, False]).iloc[0]

    summary_md = output_dir / "active_vs_rest_threshold_sweep.md"
    report_lines = [
        "# ACTIVE vs REST Threshold Sweep",
        "",
        f"- Session: `{audit['filename']}`",
        f"- Split: first `{args.train_fraction:.0%}` train, last `{1.0 - args.train_fraction:.0%}` test",
        f"- Model: `RandomForest`",
        f"- Output CSV: `{results_csv}`",
        "",
        "## Results",
        "",
        dataframe_to_markdown(
            sweep_df.loc[
                :,
                ["threshold", "accuracy", "macro_f1", "active_precision", "active_recall", "misses", "false_triggers"],
            ]
        ),
        "",
        "## Interpretation",
        "",
        f"- Best macro-F1 threshold: `{best_macro['threshold']:.1f}` with macro-F1 `{best_macro['macro_f1']:.3f}`",
        f"- Best recall threshold: `{best_recall['threshold']:.1f}` with ACTIVE recall `{best_recall['active_recall']:.3f}`",
        f"- Best balance threshold: `{best_balance['threshold']:.1f}` with macro-F1 `{best_balance['macro_f1']:.3f}`, ACTIVE recall `{best_balance['active_recall']:.3f}`, and `{int(best_balance['false_triggers'])}` false triggers",
    ]
    write_markdown(summary_md, "\n".join(report_lines))

    print(dataframe_to_markdown(
        sweep_df.loc[
            :,
            ["threshold", "accuracy", "macro_f1", "active_precision", "active_recall", "misses", "false_triggers"],
        ]
    ))
    print("\nInterpretation")
    print(f"- Best macro-F1 threshold: {best_macro['threshold']:.1f}")
    print(f"- Best recall threshold: {best_recall['threshold']:.1f}")
    print(
        f"- Best balance threshold: {best_balance['threshold']:.1f} "
        f"(macro-F1={best_balance['macro_f1']:.3f}, recall={best_balance['active_recall']:.3f}, "
        f"false_triggers={int(best_balance['false_triggers'])})"
    )
    print(f"\nWrote {results_csv}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
