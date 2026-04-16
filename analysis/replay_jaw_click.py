from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.event_utils import EventWindowConfig, JawEventConfig
    from analysis.live_jaw_click_detector import (
        ReplayEvaluationConfig,
        build_default_trigger_grid,
        build_jaw_replay_dataset,
        build_replay_score_frame,
        choose_best_config,
        evaluate_clicks_against_reference,
        export_strategy_artifacts,
        extract_reference_onsets,
        run_trigger_strategy,
        summarize_strategy_performance,
        train_jaw_click_models,
        write_best_strategy_report,
    )
    from analysis.jaw_trigger_rules import JawClickTriggerConfig
    from analysis.utils import audit_all_sessions, ensure_output_dir, write_json
else:
    from .event_utils import EventWindowConfig, JawEventConfig
    from .live_jaw_click_detector import (
        ReplayEvaluationConfig,
        build_default_trigger_grid,
        build_jaw_replay_dataset,
        build_replay_score_frame,
        choose_best_config,
        evaluate_clicks_against_reference,
        export_strategy_artifacts,
        extract_reference_onsets,
        run_trigger_strategy,
        summarize_strategy_performance,
        train_jaw_click_models,
        write_best_strategy_report,
    )
    from .jaw_trigger_rules import JawClickTriggerConfig
    from .utils import audit_all_sessions, ensure_output_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay jaw sessions and emit pseudo-live click events.")
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
    parser.add_argument("--match-pre-sec", type=float, default=ReplayEvaluationConfig().match_pre_sec)
    parser.add_argument("--match-post-sec", type=float, default=ReplayEvaluationConfig().match_post_sec)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for replay outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()

    event_config = JawEventConfig(
        smoothing_sec=args.smoothing_sec,
        onset_duration_sec=args.onset_sec,
        offset_duration_sec=args.offset_sec,
        minimum_active_duration_sec=args.minimum_active_sec,
        minimum_peak_distance_sec=args.minimum_peak_distance_sec,
        inactive_quantile=args.inactive_quantile,
        active_quantile=args.active_quantile,
        threshold_mix=args.threshold_mix,
        release_threshold_mix=args.release_threshold_mix,
    )
    window_config = EventWindowConfig(window_sec=args.window_sec, overlap=args.overlap)
    evaluation_config = ReplayEvaluationConfig(
        match_pre_sec=args.match_pre_sec,
        match_post_sec=args.match_post_sec,
    )

    dataset_bundle = build_jaw_replay_dataset(audits, event_config, window_config)
    model_bundle = train_jaw_click_models(dataset_bundle)
    score_frame = build_replay_score_frame(dataset_bundle, model_bundle)
    reference_onsets = extract_reference_onsets(dataset_bundle["event_bundle"]["interval_frame"])
    dataset_bundle["event_bundle"]["sample_frame"].to_csv(output_dir / "jaw_event_samples.csv", index=False)
    dataset_bundle["event_bundle"]["interval_frame"].to_csv(output_dir / "jaw_event_intervals.csv", index=False)
    dataset_bundle["event_bundle"]["coarse_segment_frame"].to_csv(
        output_dir / "jaw_event_segment_summary.csv",
        index=False,
    )
    dataset_bundle["event_bundle"]["threshold_frame"].to_csv(output_dir / "jaw_event_thresholds.csv", index=False)
    score_csv = output_dir / "jaw_click_replay_scores.csv"
    score_frame.to_csv(score_csv, index=False)
    reference_csv = output_dir / "jaw_click_reference_onsets.csv"
    reference_onsets.to_csv(reference_csv, index=False)

    tuning_rows = []
    for config in build_default_trigger_grid():
        click_frame = run_trigger_strategy(score_frame, config)
        summary_df, _ = evaluate_clicks_against_reference(click_frame, reference_onsets, evaluation_config)
        summary = summarize_strategy_performance(summary_df, config.strategy_name, config)
        tuning_rows.append(summary)

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        [
            "strategy_name",
            "train_weighted_event_f1",
            "train_weighted_precision",
            "test_weighted_event_f1",
        ],
        ascending=[True, False, False, False],
    )
    tuning_csv = output_dir / "jaw_click_strategy_tuning.csv"
    tuning_df.to_csv(tuning_csv, index=False)

    final_strategy_rows = []
    for strategy_name in tuning_df["strategy_name"].drop_duplicates():
        strategy_candidates = tuning_df[tuning_df["strategy_name"] == strategy_name].to_dict(orient="records")
        best_strategy_summary = choose_best_config(strategy_candidates)
        best_config = best_strategy_summary["trigger_config"]
        trigger_config = JawClickTriggerConfig(**best_config)
        click_frame = run_trigger_strategy(score_frame, trigger_config)
        summary_df, match_df = evaluate_clicks_against_reference(click_frame, reference_onsets, evaluation_config)
        strategy_summary = summarize_strategy_performance(summary_df, strategy_name, trigger_config)
        strategy_summary.update(
            export_strategy_artifacts(
                output_dir,
                strategy_name,
                score_frame,
                click_frame,
                summary_df,
                match_df,
            )
        )
        final_strategy_rows.append(strategy_summary)

    final_df = pd.DataFrame(final_strategy_rows).sort_values(
        ["test_weighted_event_f1", "test_weighted_precision", "train_weighted_event_f1"],
        ascending=[False, False, False],
    )
    final_csv = output_dir / "jaw_click_strategy_summary.csv"
    final_df.to_csv(final_csv, index=False)
    best_row = final_df.iloc[0].to_dict()

    write_best_strategy_report(
        output_dir / "jaw_click_best_strategy.md",
        output_dir / "jaw_click_best_strategy.json",
        best_row,
        final_df,
        evaluation_config,
    )
    write_json(
        output_dir / "jaw_click_replay_metadata.json",
        {
            "event_config": asdict(event_config),
            "window_config": asdict(window_config),
            "evaluation_config": asdict(evaluation_config),
            "selected_channels": model_bundle["selected_channels"],
            "excluded_channels": model_bundle["excluded_channels"],
            "score_csv": str(score_csv),
            "reference_csv": str(reference_csv),
            "tuning_csv": str(tuning_csv),
            "summary_csv": str(final_csv),
        },
    )

    print("\nJaw click replay strategy comparison")
    for _, row in final_df.iterrows():
        print(
            f"  {row['strategy_name']}: "
            f"test_f1={row['test_weighted_event_f1']:.3f} "
            f"test_precision={row['test_weighted_precision']:.3f} "
            f"test_recall={row['test_weighted_recall']:.3f} "
            f"test_clicks={int(row['test_total_clicks'])} "
            f"test_refs={int(row['test_total_references'])}"
        )
    print(f"\nBest strategy: {best_row['strategy_name']}")
    print(f"Trigger config: {best_row['trigger_config']}")
    print(f"Wrote {final_csv}")


if __name__ == "__main__":
    main()
