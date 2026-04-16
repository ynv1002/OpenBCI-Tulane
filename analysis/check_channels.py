from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import (
        FAMILY_SPECS,
        audit_all_sessions,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        split_train_test_audits,
    )
else:
    from .utils import (
        FAMILY_SPECS,
        audit_all_sessions,
        dataframe_to_markdown,
        ensure_output_dir,
        select_model_channels,
        split_train_test_audits,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check per-channel quality across the configured sessions.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for channel-quality outputs.",
    )
    return parser.parse_args()


def _channel_rows(audits: List[dict]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for audit in audits:
        for row in audit["channel_quality"]:
            rows.append(
                {
                    "family": audit["family"],
                    "session_rank": audit["session_rank"],
                    "filename": audit["filename"],
                    **row,
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "session_rank", "channel"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    audits = audit_all_sessions()
    channel_df = _channel_rows(audits)

    channel_csv = output_dir / "channel_quality.csv"
    channel_md = output_dir / "channel_quality.md"
    channel_df.to_csv(channel_csv, index=False)

    family_summary_rows = []
    lines = ["# Channel Quality", ""]
    for family_key, spec in FAMILY_SPECS.items():
        train_audits, test_audit = split_train_test_audits(audits, family_key)
        selected_channels, excluded_channels = select_model_channels(train_audits)
        family_summary_rows.append(
            {
                "family": family_key,
                "train_files": ", ".join(audit["filename"] for audit in train_audits),
                "test_file": test_audit["filename"],
                "selected_channels": ", ".join(selected_channels),
                "excluded_channels": " | ".join(f"{key}: {value}" for key, value in excluded_channels.items()),
            }
        )

        lines.extend(
            [
                f"## {spec.display_name}",
                "",
                f"- Train files: `{', '.join(audit['filename'] for audit in train_audits)}`",
                f"- Test file: `{test_audit['filename']}`",
                f"- Selected channels: `{selected_channels}`",
                f"- Excluded channels: `{' | '.join(f'{key}: {value}' for key, value in excluded_channels.items())}`",
                "",
            ]
        )

    summary_df = pd.DataFrame(family_summary_rows)
    lines.append(dataframe_to_markdown(summary_df))
    channel_md.write_text("\n".join(lines), encoding="utf-8")

    display_df = channel_df[
        [
            "family",
            "session_rank",
            "filename",
            "channel",
            "status",
            "rail_fraction",
            "dominant_value_fraction",
            "centered_std",
            "reasons",
        ]
    ]
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 120)
    print(display_df.to_string(index=False))
    print(f"\nWrote {channel_csv}")
    print(f"Wrote {channel_md}")


if __name__ == "__main__":
    main()
