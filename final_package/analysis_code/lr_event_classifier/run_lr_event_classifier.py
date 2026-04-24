from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from final_package.analysis_code.lr_event_classifier.review_lr_event_classifier import main


if __name__ == "__main__":
    main()
