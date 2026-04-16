from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, inventory_rows
else:
    from .utils import audit_all_sessions, dataframe_to_markdown, ensure_output_dir, inventory_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory the OpenBCI CSV files in the two configured folders.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for inventory outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    audits = audit_all_sessions()
    inventory_df = inventory_rows(audits)

    inventory_csv = output_dir / "data_inventory.csv"
    inventory_md = output_dir / "data_inventory.md"
    inventory_df.to_csv(inventory_csv, index=False)

    display_df = inventory_df[
        [
            "family",
            "session_rank",
            "filename",
            "parsed_date",
            "shape",
            "marker_column_exists",
            "sample_rate_used_hz",
            "usable_for_modeling",
            "top_issue",
        ]
    ].copy()
    markdown = "\n".join(
        [
            "# Data Inventory",
            "",
            "Session order uses the embedded filename date first and the run index in parentheses second.",
            "",
            dataframe_to_markdown(display_df),
        ]
    )
    inventory_md.write_text(markdown, encoding="utf-8")

    pd.set_option("display.max_colwidth", 120)
    print(display_df.to_string(index=False))
    print(f"\nWrote {inventory_csv}")
    print(f"Wrote {inventory_md}")


if __name__ == "__main__":
    main()
