"""
liveness.py -- Blink-based liveness detection (anti-spoofing).

Why this works
--------------
A photograph or a phone screen shown to the camera does not blink. We
compute the Eye-Aspect-Ratio (EAR) for each tracked face
(Soukupova & Cech, 2016):

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

where p1..p6 are the six landmarks of one eye. With the eye open the EAR
hovers around 0.25-0.35; when the eye closes it collapses below ~0.20.
Counting the dips (with hysteresis, so one noisy frame cannot fake a blink)
gives a lightweight, hardware-free proof of life.

Eye landmarks come from face_recognition.face_landmarks(), which is driven
by dlib's 68-point model shipped INSIDE the library - no extra .dat file to
download. The face_recognition import is done lazily inside update() so the
pure-math helpers stay unit-testable without dlib (see tests/test_liveness.py).
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from config import (BLINKS_REQUIRED, EAR_CLOSED_FRAMES, EAR_CLOSED_THRESH,
                    LIVENESS_TIMEOUT, SESSION_TIMEOUT)

# Public status codes returned by LivenessChecker.update()
ANALYZING = "ANALYZING"   # face seen, waiting for the blink
LIVE      = "LIVE"        # at least one blink observed -> real person
SUSPECT   = "SUSPECT"     # no blink within LIVENESS_TIMEOUT -> possible photo


def eye_aspect_ratio(eye_points) -> float:
    """EAR from six (x, y) eye-corner points. Safe against bad input."""
    p = np.asarray(eye_points, dtype=np.float64)
    if p.shape != (6, 2):
        return 1.0
    vertical = float(np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4]))
    horizontal = float(np.linalg.norm(p[0] - p[3]))
    return vertical / (2.0 * horizontal) if horizontal > 1e-6 else 1.0


def _ear_from_landmarks(landmarks: Dict[str, list]) -> float:
    """Minimum EAR of both eyes (the more-closed eye wins) or 1.0 if absent."""
    ears = [
        eye_aspect_ratio(landmarks[key])
        for key in ("left_eye", "right_eye")
        if key in landmarks and len(landmarks[key]) == 6
    ]
    return min(ears) if ears else 1.0


class LivenessChecker:
    """Per-face blink state machine.

    One instance per camera. It is keyed by an arbitrary `face_key` (the main
    app uses the recognised person's name). Call `update()` once per processed
    frame for every known face on screen; it returns the current verdict.
    Faces that disappear for more than SESSION_TIMEOUT seconds are forgotten,
    so a person who walks away and returns has to blink again.
    """

    def __init__(self) -> None:
        self._faces: Dict[str, dict] = {}

    # -- internal helpers ---------------------------------------------------
    def _purge_stale(self, now: float) -> None:
        """Forget faces that have not been seen recently."""
        stale = [k for k, s in self._faces.items()
                 if now - s["last_seen"] > SESSION_TIMEOUT]
        for key in stale:
            del self._faces[key]

    # -- public API -----------------------------------------------------------
    def reset(self, face_key: str) -> None:
        """Forget one face (e.g. when you know the scene changed)."""
        self._faces.pop(face_key, None)

    def update(self, face_key: str, frame_rgb: np.ndarray,
               box: Tuple[int, int, int, int],
               now: Optional[float] = None) -> dict:
        """Feed one detection of a face.

        Parameters
        ----------
        face_key   : stable identifier for the face (we use the person name)
        frame_rgb  : FULL-RESOLUTION RGB frame (landmarks need the pixels)
        box        : (top, right, bottom, left) in the same coordinate space
        now        : injectable timestamp (tests); defaults to time.time()

        Returns {"status": ANALYZING|LIVE|SUSPECT, "ear": float,
                 "blinks": int, "elapsed": float}
        """
        import face_recognition  # lazy import: keeps this module cheap & testable

        now = time.time() if now is None else now
        self._purge_stale(now)

        state = self._faces.get(face_key)
        if state is None:
            state = {"first_seen": now, "last_seen": now, "closed_run": 0,
                     "blinks": 0, "live": False, "last_ear": 1.0}
            self._faces[face_key] = state
        state["last_seen"] = now

        # --- landmarks on a padded crop around the face (much faster than
        #     running on the whole frame) -------------------------------------
        top, right, bottom, left = box
        fh, fw = frame_rgb.shape[:2]
        pad = max(8, (right - left) // 4)
        y0, y1 = max(0, top - pad), min(fh, bottom + pad)
        x0, x1 = max(0, left - pad), min(fw, right + pad)

        ear = 1.0
        if y1 > y0 and x1 > x0:
            crop = frame_rgb[y0:y1, x0:x1]
            inner = (top - y0, right - x0, bottom - y0, left - x0)
            try:
                marks = face_recognition.face_landmarks(crop, [inner])
                ear = _ear_from_landmarks(marks[0]) if marks else 1.0
            except Exception:
                ear = 1.0  # dlib occasionally fails on extreme crops - never crash
        state["last_ear"] = ear

        # --- blink state machine (hysteresis: N consecutive closed frames) --
        if ear < EAR_CLOSED_THRESH:
            state["closed_run"] += 1
        else:
            if state["closed_run"] >= EAR_CLOSED_FRAMES:
                state["blinks"] += 1
                if state["blinks"] >= BLINKS_REQUIRED:
                    state["live"] = True
            state["closed_run"] = 0

        elapsed = now - state["first_seen"]
        if state["live"]:
            status = LIVE
        elif (state["blinks"] == 0 and state["closed_run"] == 0
              and elapsed > LIVENESS_TIMEOUT):
            status = SUSPECT
        else:
            status = ANALYZING

        return {"status": status, "ear": round(ear, 3),
                "blinks": state["blinks"], "elapsed": elapsed}