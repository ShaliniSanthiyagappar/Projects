"""
dataset_builder.py -- Turns dataset/<PersonName>/ photos into face encodings.

For every photo it detects the largest face and stores its 128-D encoding
together with the folder name (the person's display name). Everything lands
in data/encodings.pickle, which attendance_app.py loads at startup.

Usage
-----
    python dataset_builder.py             # incremental: only NEW/CHANGED
                                          # photos are encoded (fast)
    python dataset_builder.py --force     # re-encode everything from scratch
    python dataset_builder.py --model cnn # GPU detection if you built CUDA dlib

Adding or removing photos later? Just re-run it - unchanged photos are
reused, so the rebuild is quick.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import face_recognition

from config import DATASET_DIR, ENCODINGS_FILE, IMAGE_EXTS, MODEL


def build_encodings(dataset_dir: Path,
                    out_file: Path,
                    model: str = MODEL,
                    force: bool = False) -> int:
    """Encode every dataset image. Returns 0 on success, 1 on user error."""
    if not dataset_dir.exists():
        print(f"ERROR: dataset folder not found: {dataset_dir}")
        print("Create it like this, then re-run:\n")
        print("    dataset/")
        print("    |-- Alice_Smith/")
        print("    |   |-- photo1.jpg")
        print("    |   `-- photo2.jpg")
        print("    `-- Raj_Patel/")
        print("        `-- selfie.png")
        return 1

    people_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
    if not people_dirs:
        print(f"ERROR: no person folders inside {dataset_dir}.")
        print("Create one folder per person (folder name = displayed name) "
              "and put 3-10 clear photos in each.")
        return 1

    # ---- load the previous build so unchanged photos can be reused ---------
    prev_index: Dict[str, dict] = {}      # key -> {"name", "mtime", "pos"}
    prev_encodings: list = []
    if out_file.exists() and not force:
        try:
            with open(out_file, "rb") as f:
                prev = pickle.load(f)
            prev_index = prev.get("index", {})
            prev_encodings = prev.get("encodings", [])
            if prev_encodings:
                print(f"Reusing previous build ({len(prev_encodings)} encodings); "
                      f"only new/changed photos will be encoded. "
                      f"Use --force to rebuild everything.\n")
        except Exception:
            print("Could not read the previous encodings file - rebuilding "
                  "from scratch.\n")

    # key "Person/file.jpg" -> (person_name, encoding, mtime)
    kept: Dict[str, Tuple[str, np.ndarray, float]] = {}
    for key, meta in prev_index.items():
        img_path = dataset_dir / key
        try:
            if img_path.exists() and abs(img_path.stat().st_mtime - meta["mtime"]) < 1e-6:
                pos = meta.get("pos", -1)
                if 0 <= pos < len(prev_encodings):
                    kept[key] = (meta["name"], prev_encodings[pos], meta["mtime"])
        except OSError:
            continue  # unreadable file - it will simply be re-encoded or dropped

    # ---- encode every person's photos ---------------------------------------
    print(f"Scanning {len(people_dirs)} person folder(s) in {dataset_dir}\n")
    for person_dir in people_dirs:
        person = person_dir.name.strip()
        images = sorted(p for p in person_dir.iterdir()
                        if p.suffix.lower() in IMAGE_EXTS)
        if not images:
            print(f"  [warn] {person_dir.name}/ has no images - skipped")
            continue

        added = reused = failed = 0
        for img_path in images:
            key = f"{person}/{img_path.name}"
            if key in kept:                       # unchanged since last build
                reused += 1
                continue
            try:
                image = face_recognition.load_image_file(str(img_path))
                boxes = face_recognition.face_locations(image, model=model)
                if not boxes:
                    print(f"  [warn] {person}/{img_path.name}: no face found - skipped")
                    failed += 1
                    continue
                # keep only the LARGEST face = the subject of the photo
                top, right, bottom, left = max(
                    boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                enc = face_recognition.face_encodings(
                    image, [(top, right, bottom, left)])[0]
                kept[key] = (person, enc, img_path.stat().st_mtime)
                added += 1
            except Exception as exc:
                print(f"  [warn] {person}/{img_path.name}: {exc} - skipped")
                failed += 1

        line = f"  {person:<24} +{added} encoded, {reused} reused"
        if failed:
            line += f", {failed} skipped"
        print(line)

    if not kept:
        print("\nERROR: not a single face could be encoded. Check your photos.")
        return 1

    # ---- write the combined encodings file -----------------------------------
    names: list = []
    encodings: list = []
    index: Dict[str, dict] = {}
    for pos, key in enumerate(sorted(kept)):
        person, enc, mtime = kept[key]
        names.append(person)
        encodings.append(np.asarray(enc, dtype=np.float64))
        index[key] = {"name": person, "mtime": mtime, "pos": pos}

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump({"names": names, "encodings": encodings, "index": index}, f)

    people = sorted(set(names))
    print(f"\nDone: {len(encodings)} encodings for {len(people)} person(s)")
    print(f"  People : {', '.join(people)}")
    print(f"  Saved  : {out_file}")
    print("\nNext step:  python attendance_app.py")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build face encodings from dataset/<PersonName>/ photos.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR,
                        help=f"dataset folder (default: {DATASET_DIR})")
    parser.add_argument("--output", type=Path, default=ENCODINGS_FILE,
                        help=f"output pickle (default: {ENCODINGS_FILE})")
    parser.add_argument("--model", choices=("hog", "cnn"), default=MODEL,
                        help='face detection model (default: "%s")' % MODEL)
    parser.add_argument("--force", action="store_true",
                        help="re-encode ALL photos, ignore the previous build")
    args = parser.parse_args()

    print("=" * 62)
    print(" Dataset Builder - Face Recognition Attendance System")
    print("=" * 62)
    return build_encodings(args.dataset, args.output, args.model, args.force)


if __name__ == "__main__":
    sys.exit(main())