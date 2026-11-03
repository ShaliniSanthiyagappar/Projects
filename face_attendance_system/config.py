"""
config.py -- Central settings for the Face Recognition Attendance System.

Every tunable knob lives here so you never have to dig through application
logic. A tiny built-in `.env` loader (zero dependencies) lets you keep
machine-specific overrides out of the source code: create a `.env` file next
to this script with lines like `FRS_MODEL=cnn` and they are picked up
automatically. Real environment variables always win over `.env` values.
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Optional .env support
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE pairs from `path` into os.environ (existing env wins)."""
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass  # a broken .env must never crash the application


BASE_DIR = Path(__file__).resolve().parent
_load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------------------------
# Paths (directories are created automatically on first run where needed)
# ---------------------------------------------------------------------------
DATASET_DIR    = BASE_DIR / "dataset"   # dataset/<PersonName>/*.jpg  (you add these)
DATA_DIR       = BASE_DIR / "data"      # encodings.pickle, attendance.db, ...
ENCODINGS_FILE = DATA_DIR / "encodings.pickle"
DB_FILE        = DATA_DIR / "attendance.db"
EXPORT_DIR     = BASE_DIR / "exports"   # daily CSV files are written here

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}  # accepted dataset image types


# ---------------------------------------------------------------------------
# Camera / performance
# ---------------------------------------------------------------------------
CAMERA_INDEX          = 0     # 0 = built-in / first webcam, 1 = second, ...
FRAME_WIDTH           = 1280  # capture width requested from the camera
PROCESS_WIDTH         = 640   # recognition runs on a copy downscaled to this width
PROCESS_EVERY_N_FRAME = 2     # run the heavy recognition every Nth frame (1 = every frame)
MIRROR_PREVIEW        = True  # horizontal flip for a natural "selfie" view


# ---------------------------------------------------------------------------
# Recognition
# ---------------------------------------------------------------------------
MODEL           = os.getenv("FRS_MODEL", "hog")  # "hog" = fast CPU | "cnn" = NVIDIA GPU
TOLERANCE       = 0.50   # max face distance to accept a match (0.45 strict .. 0.60 loose)
MIN_FACE_HEIGHT = 60     # ignore faces smaller than this (px, in the process frame)


# ---------------------------------------------------------------------------
# Liveness detection (blink-based anti-spoofing) -- see liveness.py for the math
# ---------------------------------------------------------------------------
REQUIRE_LIVENESS  = True  # False = mark attendance on first sight (testing only!)
EAR_CLOSED_THRESH = 0.21  # Eye-Aspect-Ratio below this counts as "eye closed"
EAR_CLOSED_FRAMES = 2     # consecutive "closed" frames that count as one blink
BLINKS_REQUIRED   = 1     # blinks needed before a face is accepted as live
LIVENESS_TIMEOUT  = 12.0  # s a face may stay unverified before the "photo?" flag
SESSION_TIMEOUT   = 20.0  # s without a sighting before a face's state is forgotten


# ---------------------------------------------------------------------------
# Attendance rules / unknown faces
# ---------------------------------------------------------------------------
LATE_AFTER           = "09:30"  # dashboard flags later check-ins as "Late"
LOG_UNKNOWN          = True     # write timestamped rows for unregistered faces
UNKNOWN_LOG_COOLDOWN = 30.0     # min seconds between two unknown-sighting rows


# ---------------------------------------------------------------------------
# On-screen colours (OpenCV uses BGR order)
# ---------------------------------------------------------------------------
COLOR_LIVE      = (80, 220, 80)    # green  - live person / attendance recorded
COLOR_ANALYZING = (30, 210, 235)   # yellow - waiting for the blink
COLOR_SPOOF     = (30, 120, 255)   # orange - no blink within the timeout
COLOR_UNKNOWN   = (60, 60, 230)    # red    - unregistered face