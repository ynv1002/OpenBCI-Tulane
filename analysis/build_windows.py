from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from analysis.utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        FAMILY_SPECS,
        audit_all_sessions,
        build_family_windows,
        ensure_output_dir,
    )
else:
    from .utils import (
        DEFAULT_OVERLAP,
        DEFAULT_WINDOW_SEC,
        FAMILY_SPECS,
        audit_all_sessions,
        build_family_windows,
        ensure_output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build window-level feature tables for the configured sessions.")
    parser.add_argument(
        "--family",
        choices=["all", *FAMILY_SPECS.keys()],
        default="all",
        help="Which experiment family to process.",
    )
    parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC, help="Window length in seconds.")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="Fractional window overlap.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for window outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    audits = audit_all_sessions()

    family_keys = list(FAMILY_SPECS) if args.family == "all" else [args.family]
    for family_key in family_keys:
        bundle = build_family_windows(audits, family_key, window_sec=args.window_sec, overlap=args.overlap)
        combined_df = pd.concat([bundle["train_windows"], bundle["test_windows"]], ignore_index=True)

        windows_csv = output_dir / f"window_features_{family_key}.csv"
        arrays_npz = output_dir / f"window_arrays_{family_key}.npz"
        combined_df.to_csv(windows_csv, index=False)

        np.savez(
            arrays_npz,
            X=combined_df[bundle["feature_columns"]].to_numpy(dtype=float),
            y=combined_df["label"].to_numpy(),
            feature_names=np.array(bundle["feature_columns"], dtype=object),
            filenames=combined_df["filename"].to_numpy(),
            split_role=combined_df["split_role"].to_numpy(),
        )

        print(f"\n[{family_key}] selected channels: {bundle['selected_channels']}")
        print(
            combined_df.groupby(["split_role", "label"]).size().reset_index(name="window_count").to_string(index=False)
        )
        print(f"Wrote {windows_csv}")
        print(f"Wrote {arrays_npz}")


if __name__ == "__main__":
    main()
