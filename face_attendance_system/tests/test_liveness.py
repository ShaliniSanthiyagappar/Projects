"""
Unit tests for the pure-math parts of liveness.py (no dlib needed).

Run from the project root:
    python tests/test_liveness.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liveness import EAR_CLOSED_THRESH, eye_aspect_ratio  # noqa: E402


def main() -> int:
    # synthetic OPEN eye: vertical spread of 1.5 px on each side, width 10 px
    # EAR = (3 + 3) / (2 * 10) = 0.30  ->  above the closed threshold
    open_ear = eye_aspect_ratio([(0, 0), (3, -1.5), (7, -1.5), (10, 0), (7, 1.5), (3, 1.5)])
    assert abs(open_ear - 0.30) < 1e-9, open_ear
    assert open_ear > EAR_CLOSED_THRESH
    print(f"PASS  open eye  EAR = {open_ear:.3f}  (threshold {EAR_CLOSED_THRESH})")

    # synthetic CLOSED eye: all vertical spread gone
    # EAR = 0  ->  far below the threshold
    closed_ear = eye_aspect_ratio([(0, 0), (3, 0), (7, 0), (10, 0), (7, 0), (3, 0)])
    assert closed_ear < EAR_CLOSED_THRESH
    print(f"PASS  closed eye EAR = {closed_ear:.3f}")

    # malformed input must not crash - falls back to "wide open" (1.0)
    assert eye_aspect_ratio([(0, 0)]) == 1.0
    assert eye_aspect_ratio([(0, 0)] * 6 + [(1, 1)]) == 1.0  # wrong point count
    print("PASS  malformed input handled safely")

    print("\nAll liveness tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())