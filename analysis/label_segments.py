from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import audit_all_sessions, ensure_output_dir
else:
    from .utils import audit_all_sessions, ensure_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert marker runs into labeled baseline/rest/active segments.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for segment outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    audits = audit_all_sessions()
    rows: List[dict] = []
    for audit in audits:
        for segment in audit["segments"]:
            rows.append(
                {
                    "family": audit["family"],
                    "filename": audit["filename"],
                    "file_path": str(audit["file_path"]),
                    "parsed_date": audit["parsed_date"],
                    "session_rank": audit["session_rank"],
                    **segment,
                }
            )

    segments_df = pd.DataFrame(rows).sort_values(
        ["family", "session_rank", "segment_index"]
    ).reset_index(drop=True)
    summary_df = (
        segments_df.groupby(["family", "label"])
        .agg(segment_count=("label", "size"), total_duration_sec=("duration_sec", "sum"))
        .reset_index()
    )

    segments_csv = output_dir / "segments.csv"
    summary_csv = output_dir / "segments_summary.csv"
    segments_df.to_csv(segments_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(segments_df.to_string(index=False, max_rows=200))
    print(f"\nWrote {segments_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
