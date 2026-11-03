"""
recognizer.py -- Face detection + identification wrapper.

Isolates every call into the face_recognition library behind one small class,
so the rest of the app never touches dlib directly.

Pipeline inside recognize()
---------------------------
1. face_locations()  -> bounding boxes of ALL faces in the frame (HOG or CNN)
2. face_encodings()  -> one 128-D embedding per detected face
3. face_distance()   -> Euclidean distance of each embedding against every
                        known embedding; the closest match wins, and it is
                        only accepted when distance <= TOLERANCE
Boxes use face_recognition's (top, right, bottom, left) convention.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

import face_recognition

from config import ENCODINGS_FILE, MIN_FACE_HEIGHT, MODEL, TOLERANCE

UNKNOWN = "Unknown"


class FaceRecognizer:
    """Loads data/encodings.pickle once, then matches frames against it."""

    def __init__(self,
                 encodings_file: Path = ENCODINGS_FILE,
                 model: str = MODEL,
                 tolerance: float = TOLERANCE) -> None:
        encodings_file = Path(encodings_file)
        if not encodings_file.exists():
            raise FileNotFoundError(
                f"No encodings file found at {encodings_file}.\n"
                "Run  python dataset_builder.py  first - it turns your "
                "dataset/<PersonName>/ photos into this file."
            )
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)

        self._known_encodings = [np.asarray(e, dtype=np.float64)
                                 for e in data.get("encodings", [])]
        self._known_names: List[str] = list(data.get("names", []))

        if not self._known_encodings:
            raise ValueError(
                "encodings.pickle contains no faces - add photos under "
                "dataset/<PersonName>/ and re-run dataset_builder.py."
            )

        self.model = model          # "hog" (fast CPU) or "cnn" (GPU)
        self.tolerance = tolerance  # lower = stricter matching

    # ------------------------------------------------------------------ props
    @property
    def people(self) -> List[str]:
        """Unique names of all enrolled people."""
        return sorted(set(self._known_names))

    @property
    def encoding_count(self) -> int:
        """Total number of stored face encodings (photos encoded)."""
        return len(self._known_encodings)

    # ------------------------------------------------------------------ main
    def recognize(self, frame_rgb: np.ndarray) -> List[Dict]:
        """Detect all faces in an RGB frame and match them against the roster.

        Returns a list of dicts: {"name": str, "distance": float, "box": tuple}
        where box is (top, right, bottom, left) in THIS frame's pixels.
        Faces smaller than MIN_FACE_HEIGHT are ignored (too far from camera).
        """
        boxes = face_recognition.face_locations(frame_rgb, model=self.model)
        boxes = [b for b in boxes if (b[2] - b[0]) >= MIN_FACE_HEIGHT]
        if not boxes:
            return []

        encodings = face_recognition.face_encodings(frame_rgb, boxes)

        results: List[Dict] = []
        for box, encoding in zip(boxes, encodings):
            name, distance = UNKNOWN, float("inf")
            if self._known_encodings:
                distances = face_recognition.face_distance(
                    self._known_encodings, encoding)
                best = int(np.argmin(distances))
                if distances[best] <= self.tolerance:
                    name, distance = self._known_names[best], float(distances[best])
            results.append({"name": name,
                            "distance": round(distance, 3),
                            "box": box})
        return results