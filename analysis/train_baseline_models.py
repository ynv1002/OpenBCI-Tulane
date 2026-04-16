from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        FAMILY_SPECS,
        LABEL_BASELINE,
        LABEL_REST,
        audit_all_sessions,
        build_family_windows,
        ensure_output_dir,
        family_spec,
        write_json,
    )
else:
    from .utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        FAMILY_SPECS,
        LABEL_BASELINE,
        LABEL_REST,
        audit_all_sessions,
        build_family_windows,
        ensure_output_dir,
        family_spec,
        write_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train first-pass offline baseline models on the configured sessions.")
    parser.add_argument(
        "--family",
        choices=["all", *FAMILY_SPECS.keys()],
        default="all",
        help="Which experiment family to process.",
    )
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC, help="Window length in seconds.")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="Fractional window overlap.")
    parser.add_argument(
        "--skip-jaw-binary",
        action="store_true",
        help="Skip the optional jaw clench-vs-non-clench branch.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for model outputs.",
    )
    return parser.parse_args()


def model_bank() -> Dict[str, object]:
    return {
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }


def label_order_for_family(family_key: str) -> List[str]:
    spec = family_spec(family_key)
    return [LABEL_BASELINE, LABEL_REST, spec.movement1_label, spec.movement2_label]


def binary_label_order() -> List[str]:
    return ["NON_CLENCH", "CLENCH"]


def _class_counts(frame: pd.DataFrame) -> Dict[str, int]:
    counts = frame["label"].value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def _fit_and_score(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    labels: List[str],
) -> Dict[str, dict]:
    X_train = train_df[feature_columns].to_numpy(dtype=float)
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    results: Dict[str, dict] = {}
    for model_name, model in model_bank().items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = pd.DataFrame(
                confusion_matrix(y_test, y_pred, labels=labels),
                index=[f"true_{label}" for label in labels],
                columns=[f"pred_{label}" for label in labels],
            )
            results[model_name] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "confusion_matrix": cm,
            }
        except Exception as exc:
            results[model_name] = {"error": str(exc)}
    return results


def _render_report(
    family_key: str,
    task_name: str,
    bundle: Dict[str, object],
    labels: List[str],
    results: Dict[str, dict],
    output_dir: Path,
) -> Dict[str, object]:
    train_df = bundle["train_windows"]
    test_df = bundle["test_windows"]
    feature_columns = bundle["feature_columns"]

    report = {
        "family": family_key,
        "task_name": task_name,
        "train_files": [audit["filename"] for audit in bundle["train_audits"]],
        "test_file": bundle["test_audit"]["filename"],
        "selected_channels": bundle["selected_channels"],
        "excluded_channels": bundle["excluded_channels"],
        "window_sec": bundle["window_sec"],
        "overlap": bundle["overlap"],
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "train_class_counts": _class_counts(train_df),
        "test_class_counts": _class_counts(test_df),
        "models": {},
    }

    for model_name, result in results.items():
        if "error" in result:
            report["models"][model_name] = result
            continue

        confusion_csv = output_dir / f"confusion_{family_key}_{task_name}_{model_name}.csv"
        result["confusion_matrix"].to_csv(confusion_csv)
        report["models"][model_name] = {
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "confusion_matrix_csv": str(confusion_csv),
        }

    valid_models = {
        model_name: metrics
        for model_name, metrics in report["models"].items()
        if "macro_f1" in metrics
    }
    if valid_models:
        best_model_name, best_metrics = max(
            valid_models.items(), key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"])
        )
        report["best_model"] = {"name": best_model_name, **best_metrics}
    else:
        report["best_model"] = None

    summary_json = output_dir / f"summary_{family_key}_{task_name}.json"
    write_json(summary_json, report)

    lines = [
        f"# Baseline Report: {family_key} | {task_name}",
        "",
        f"- Train files: `{', '.join(report['train_files'])}`",
        f"- Test file: `{report['test_file']}`",
        f"- Selected channels: `{report['selected_channels']}`",
        f"- Excluded channels: `{' | '.join(f'{key}: {value}' for key, value in report['excluded_channels'].items())}`",
        f"- Window setup: `{report['window_sec']:.2f} s`, overlap `{report['overlap']:.2f}`",
        f"- Feature count: `{report['feature_count']}`",
        f"- Train class counts: `{report['train_class_counts']}`",
        f"- Test class counts: `{report['test_class_counts']}`",
        "",
    ]
    for model_name, metrics in report["models"].items():
        if "error" in metrics:
            lines.append(f"- {model_name}: ERROR `{metrics['error']}`")
        else:
            lines.append(
                f"- {model_name}: accuracy `{metrics['accuracy']:.3f}`, macro-F1 `{metrics['macro_f1']:.3f}`"
            )

    summary_md = output_dir / f"summary_{family_key}_{task_name}.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    report["summary_json"] = str(summary_json)
    report["summary_md"] = str(summary_md)
    return report


def _convert_to_jaw_binary(bundle: Dict[str, object]) -> Dict[str, object]:
    def remap(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["label"] = out["label"].map(
            {
                LABEL_BASELINE: "NON_CLENCH",
                LABEL_REST: "NON_CLENCH",
                family_spec("jaw").movement1_label: "CLENCH",
                family_spec("jaw").movement2_label: "CLENCH",
            }
        )
        return out

    return {**bundle, "train_windows": remap(bundle["train_windows"]), "test_windows": remap(bundle["test_windows"])}


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()

    family_keys = list(FAMILY_SPECS) if args.family == "all" else [args.family]
    overall_reports = []

    for family_key in family_keys:
        bundle = build_family_windows(audits, family_key, window_sec=args.window_sec, overlap=args.overlap)
        labels = label_order_for_family(family_key)
        results = _fit_and_score(bundle["train_windows"], bundle["test_windows"], bundle["feature_columns"], labels)
        report = _render_report(family_key, "multiclass", bundle, labels, results, output_dir)
        overall_reports.append(report)

        print(f"\n[{family_key} | multiclass]")
        for model_name, metrics in report["models"].items():
            if "error" in metrics:
                print(f"  {model_name}: ERROR {metrics['error']}")
            else:
                print(
                    f"  {model_name}: accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"
                )

        if family_key == "jaw" and not args.skip_jaw_binary:
            binary_bundle = _convert_to_jaw_binary(bundle)
            binary_results = _fit_and_score(
                binary_bundle["train_windows"],
                binary_bundle["test_windows"],
                binary_bundle["feature_columns"],
                binary_label_order(),
            )
            binary_report = _render_report(
                family_key,
                "binary_clench_vs_nonclench",
                binary_bundle,
                binary_label_order(),
                binary_results,
                output_dir,
            )
            overall_reports.append(binary_report)

            print(f"\n[{family_key} | binary_clench_vs_nonclench]")
            for model_name, metrics in binary_report["models"].items():
                if "error" in metrics:
                    print(f"  {model_name}: ERROR {metrics['error']}")
                else:
                    print(
                        f"  {model_name}: accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"
                    )

    write_json(output_dir / "baseline_reports.json", {"reports": overall_reports})


if __name__ == "__main__":
    main()
