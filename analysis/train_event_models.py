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
    from analysis.event_utils import (
        EventWindowConfig,
        JawEventConfig,
        build_event_windows,
        build_jaw_event_label_bundle,
        remap_event_labels,
        task_label_order,
    )
    from analysis.utils import audit_all_sessions, ensure_output_dir, write_json
else:
    from .event_utils import (
        EventWindowConfig,
        JawEventConfig,
        build_event_windows,
        build_jaw_event_label_bundle,
        remap_event_labels,
        task_label_order,
    )
    from .utils import audit_all_sessions, ensure_output_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train jaw event-state models.")
    parser.add_argument("--window-sec", type=float, default=EventWindowConfig().window_sec)
    parser.add_argument("--overlap", type=float, default=EventWindowConfig().overlap)
    parser.add_argument("--smoothing-sec", type=float, default=JawEventConfig().smoothing_sec)
    parser.add_argument("--onset-sec", type=float, default=JawEventConfig().onset_duration_sec)
    parser.add_argument("--offset-sec", type=float, default=JawEventConfig().offset_duration_sec)
    parser.add_argument(
        "--minimum-active-sec",
        type=float,
        default=JawEventConfig().minimum_active_duration_sec,
    )
    parser.add_argument(
        "--minimum-peak-distance-sec",
        type=float,
        default=JawEventConfig().minimum_peak_distance_sec,
    )
    parser.add_argument("--inactive-quantile", type=float, default=JawEventConfig().inactive_quantile)
    parser.add_argument("--active-quantile", type=float, default=JawEventConfig().active_quantile)
    parser.add_argument("--threshold-mix", type=float, default=JawEventConfig().threshold_mix)
    parser.add_argument(
        "--release-threshold-mix",
        type=float,
        default=JawEventConfig().release_threshold_mix,
    )
    parser.add_argument(
        "--minimum-reference-gap-uv",
        type=float,
        default=JawEventConfig().minimum_reference_gap_uv,
    )
    parser.add_argument(
        "--peak-prominence-scale",
        type=float,
        default=JawEventConfig().peak_prominence_scale,
    )
    parser.add_argument(
        "--minimum-peak-prominence-uv",
        type=float,
        default=JawEventConfig().minimum_peak_prominence_uv,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for event-model outputs.",
    )
    return parser.parse_args()


def build_event_config(args: argparse.Namespace) -> JawEventConfig:
    return JawEventConfig(
        smoothing_sec=args.smoothing_sec,
        onset_duration_sec=args.onset_sec,
        offset_duration_sec=args.offset_sec,
        minimum_active_duration_sec=args.minimum_active_sec,
        minimum_peak_distance_sec=args.minimum_peak_distance_sec,
        inactive_quantile=args.inactive_quantile,
        active_quantile=args.active_quantile,
        threshold_mix=args.threshold_mix,
        release_threshold_mix=args.release_threshold_mix,
        minimum_reference_gap_uv=args.minimum_reference_gap_uv,
        peak_prominence_scale=args.peak_prominence_scale,
        minimum_peak_prominence_uv=args.minimum_peak_prominence_uv,
    )


def model_bank() -> Dict[str, object]:
    return {
        "LDA": Pipeline([("scaler", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
        "LogisticRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=3000, class_weight="balanced"))]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
        ),
    }


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

    outputs: Dict[str, dict] = {}
    for model_name, model in model_bank().items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            cm_df = pd.DataFrame(
                cm,
                index=[f"true_{label}" for label in labels],
                columns=[f"pred_{label}" for label in labels],
            )
            per_class_recall = {}
            for index, label in enumerate(labels):
                denom = float(cm[index].sum())
                per_class_recall[label] = float(cm[index, index] / denom) if denom else 0.0
            outputs[model_name] = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "confusion_matrix": cm_df,
                "per_class_recall": per_class_recall,
            }
        except Exception as exc:
            outputs[model_name] = {"error": str(exc)}
    return outputs


def _render_report(
    task_name: str,
    bundle: Dict[str, object],
    event_bundle: Dict[str, object],
    feature_columns: List[str],
    results: Dict[str, dict],
    output_dir: Path,
) -> Dict[str, object]:
    labels = task_label_order(task_name)
    train_df = bundle["train_df"]
    test_df = bundle["test_df"]

    report = {
        "task_name": task_name,
        "train_files": [audit["filename"] for audit in event_bundle["train_audits"]],
        "test_file": event_bundle["test_audit"]["filename"],
        "selected_channels": event_bundle["selected_channels"],
        "excluded_channels": event_bundle["excluded_channels"],
        "window_sec": bundle["window_sec"],
        "overlap": bundle["overlap"],
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "train_class_counts": train_df["label"].value_counts().sort_index().to_dict(),
        "test_class_counts": test_df["label"].value_counts().sort_index().to_dict(),
        "models": {},
    }

    for model_name, metrics in results.items():
        if "error" in metrics:
            report["models"][model_name] = metrics
            continue
        confusion_csv = output_dir / f"confusion_{task_name}_{model_name}.csv"
        metrics["confusion_matrix"].to_csv(confusion_csv)
        report["models"][model_name] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_recall": metrics["per_class_recall"],
            "confusion_matrix_csv": str(confusion_csv),
        }

    valid = {name: item for name, item in report["models"].items() if "macro_f1" in item}
    if valid:
        best_name, best_metrics = max(valid.items(), key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]))
        easiest_state = max(best_metrics["per_class_recall"].items(), key=lambda item: item[1])
        hardest_state = min(best_metrics["per_class_recall"].items(), key=lambda item: item[1])
        report["best_model"] = {
            "name": best_name,
            **best_metrics,
            "easiest_state": {"label": easiest_state[0], "recall": easiest_state[1]},
            "hardest_state": {"label": hardest_state[0], "recall": hardest_state[1]},
        }
    else:
        report["best_model"] = None

    summary_json = output_dir / f"summary_{task_name}.json"
    summary_md = output_dir / f"summary_{task_name}.md"
    write_json(summary_json, report)

    lines = [
        f"# Event Model Report: {task_name}",
        "",
        f"- Train files: `{', '.join(report['train_files'])}`",
        f"- Test file: `{report['test_file']}`",
        f"- Selected channels: `{report['selected_channels']}`",
        f"- Excluded channels: `{' | '.join(f'{key}: {value}' for key, value in report['excluded_channels'].items())}`",
        f"- Window setup: `{report['window_sec']:.3f} s`, overlap `{report['overlap']:.2f}`",
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
                f"- {model_name}: accuracy `{metrics['accuracy']:.3f}`, macro-F1 `{metrics['macro_f1']:.3f}`, per-class recall `{metrics['per_class_recall']}`"
            )
    if report["best_model"] is not None:
        lines.extend(
            [
                "",
                f"- Best model: `{report['best_model']['name']}`",
                f"- Easiest state: `{report['best_model']['easiest_state']}`",
                f"- Hardest state: `{report['best_model']['hardest_state']}`",
            ]
        )
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    report["summary_json"] = str(summary_json)
    report["summary_md"] = str(summary_md)
    return report


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()
    event_config = build_event_config(args)
    event_bundle = build_jaw_event_label_bundle(audits, event_config)

    window_bundle = build_event_windows(
        event_bundle["sample_frame"],
        event_bundle["signal_columns"],
        EventWindowConfig(window_sec=args.window_sec, overlap=args.overlap),
    )
    window_df = window_bundle["window_frame"]
    feature_columns = window_bundle["feature_columns"]
    window_csv = output_dir / "jaw_event_windows.csv"
    window_df.to_csv(window_csv, index=False)

    reports = []
    for task_name in ["jaw_4state", "clench_vs_nonclench", "active_vs_other"]:
        task_df = remap_event_labels(window_df, task_name)
        train_df = task_df[task_df["split_role"] == "train"].reset_index(drop=True)
        test_df = task_df[task_df["split_role"] == "test"].reset_index(drop=True)
        labels = task_label_order(task_name)
        results = _fit_and_score(train_df, test_df, feature_columns, labels)
        report = _render_report(
            task_name,
            {
                "train_df": train_df,
                "test_df": test_df,
                "window_sec": args.window_sec,
                "overlap": args.overlap,
            },
            event_bundle,
            feature_columns,
            results,
            output_dir,
        )
        reports.append(report)
        print(f"\n[{task_name}]")
        for model_name, metrics in report["models"].items():
            if "error" in metrics:
                print(f"  {model_name}: ERROR {metrics['error']}")
            else:
                print(
                    f"  {model_name}: accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}"
                )

    write_json(
        output_dir / "jaw_event_model_reports.json",
        {
            "selected_channels": event_bundle["selected_channels"],
            "excluded_channels": event_bundle["excluded_channels"],
            "feature_columns": feature_columns,
            "window_csv": str(window_csv),
            "reports": reports,
        },
    )
    print(f"\nWrote {window_csv}")


if __name__ == "__main__":
    main()
