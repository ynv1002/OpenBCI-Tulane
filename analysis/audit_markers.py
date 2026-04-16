from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_json
else:
    from .utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit marker structure and inferred session timing.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for marker audit outputs.",
    )
    return parser.parse_args()


def _marker_summary_rows(audits: List[dict]) -> pd.DataFrame:
    rows = []
    for audit in audits:
        protocol = audit["protocol_summary"]
        rows.append(
            {
                "family": audit["family"],
                "session_rank": audit["session_rank"],
                "filename": audit["filename"],
                "movement1_count": protocol["movement1"]["count"],
                "movement2_count": protocol["movement2"]["count"],
                "baseline_median_sec": round(protocol["baseline"].get("median_duration_sec", 0.0), 2)
                if protocol["baseline"]["count"]
                else None,
                "movement1_median_sec": round(protocol["movement1"].get("median_duration_sec", 0.0), 2)
                if protocol["movement1"]["count"]
                else None,
                "movement2_median_sec": round(protocol["movement2"].get("median_duration_sec", 0.0), 2)
                if protocol["movement2"]["count"]
                else None,
                "marker_counts": audit["marker_counts"],
                "pairing_issue_count": len(audit["protocol_summary"]["protocol_issues"])
                + len(audit["issues"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "session_rank"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    audits = audit_all_sessions()
    summary_df = _marker_summary_rows(audits)

    summary_csv = output_dir / "marker_audit_summary.csv"
    summary_md = output_dir / "marker_audit.md"
    summary_json = output_dir / "marker_audit.json"

    summary_df.to_csv(summary_csv, index=False)
    write_json(summary_json, {"audits": audits})

    lines = ["# Marker Audit", ""]
    lines.append(dataframe_to_markdown(summary_df))
    for audit in audits:
        protocol = audit["protocol_summary"]
        lines.extend(
            [
                "",
                f"## {audit['family']} | {audit['filename']}",
                "",
                f"- File: `{audit['file_path']}`",
                f"- Shape: `{tuple(audit['shape'])}`",
                f"- Sample-rate used: `{audit['sampling']['fs_used_hz']:.3f} Hz`",
                f"- Marker counts: `{audit['marker_counts']}`",
                f"- Movement 1 intervals: `{protocol['movement1']['count']}`",
                f"- Movement 2 intervals: `{protocol['movement2']['count']}`",
                f"- Baseline median duration: `{protocol['baseline'].get('median_duration_sec', 0.0):.2f} s`"
                if protocol["baseline"]["count"]
                else "- Baseline median duration: `n/a`",
                f"- Movement 1 median duration: `{protocol['movement1'].get('median_duration_sec', 0.0):.2f} s`"
                if protocol["movement1"]["count"]
                else "- Movement 1 median duration: `n/a`",
                f"- Movement 2 median duration: `{protocol['movement2'].get('median_duration_sec', 0.0):.2f} s`"
                if protocol["movement2"]["count"]
                else "- Movement 2 median duration: `n/a`",
                f"- Sampling notes: `{' | '.join(audit['sampling']['notes'])}`",
                f"- Issues: `{' | '.join(audit['issues'])}`" if audit["issues"] else "- Issues: `none`",
            ]
        )

    summary_md.write_text("\n".join(lines), encoding="utf-8")

    pd.set_option("display.max_colwidth", 120)
    print(summary_df.to_string(index=False))
    print(f"\nWrote {summary_csv}")
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_md}")


if __name__ == "__main__":
    main()
