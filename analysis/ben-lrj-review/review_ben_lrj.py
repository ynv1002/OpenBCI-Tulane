from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lrj_review_core import LRJReviewConfig, build_arg_parser, run_lrj_review


CONFIG = LRJReviewConfig(
    review_title="Ben LRJ Review",
    console_title="Ben LRJ review",
    plot_title="Ben LRJ review",
    cli_description="One-off Ben LRJ structural audit and trial-count review.",
    csv_help="Path to Ben-LRJ(1-6)-4:9.csv.",
    default_output_dir=Path(__file__).resolve().parent / "outputs",
)


def parse_args():
    return build_arg_parser(CONFIG).parse_args()


def run_review(
    csv_path: Path,
    output_dir: Path,
    fs_hz: float,
    make_plot: bool,
):
    return run_lrj_review(
        config=CONFIG,
        csv_path=csv_path,
        output_dir=output_dir,
        fs_hz=fs_hz,
        make_plot=make_plot,
    )


def main() -> None:
    args = parse_args()
    run_review(
        csv_path=args.csv,
        output_dir=args.output_dir,
        fs_hz=float(args.fs),
        make_plot=not bool(args.no_plot),
    )


if __name__ == "__main__":
    main()
