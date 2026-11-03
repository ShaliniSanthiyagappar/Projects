"""
attendance_app.py -- MAIN ENTRY POINT.

Real-time face-recognition attendance for one classroom / office webcam.

Pipeline (per frame)
--------------------
1.  grab a frame from the webcam (optionally mirrored for a selfie view)
2.  every Nth frame: downscale -> detect + recognise ALL faces (HOG default)
3.  for every face on screen:
      unknown   -> red box + throttled row in the unknown_sightings table
      known     -> blink-based liveness check (config.REQUIRE_LIVENESS)
                   LIVE      -> green box, attendance recorded (once per day)
                   ANALYZING -> yellow box, "blink to verify"
                   SUSPECT   -> orange box, "no blink - photo?"
4.  HUD: FPS, model, enrolled people, today's head-count
5.  on exit: today's attendance is exported to exports/attendance_<date>.csv

Usage
-----
    python attendance_app.py                  # defaults from config.py
    python attendance_app.py --no-liveness    # mark on first sight (testing)
    python attendance_app.py --camera 1       # second webcam
    python attendance_app.py --model cnn      # NVIDIA GPU (CUDA dlib build)

Keys:  q / ESC = quit    |    e = export today's CSV immediately
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime

import cv2

import config
import database
from liveness import ANALYZING, LIVE, SUSPECT, LivenessChecker
from recognizer import FaceRecognizer

_FONT = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _label(frame, text: str, x: int, y: int, color) -> None:
    """Filled colour chip with dark text, sitting on top of a face box."""
    (tw, th), base = cv2.getTextSize(text, _FONT, 0.55, 1)
    y_top = max(0, y - th - base - 8)
    cv2.rectangle(frame, (x, y_top), (x + tw + 8, y_top + th + base + 8),
                  color, -1)
    cv2.putText(frame, text, (x + 4, y_top + th + 4), _FONT, 0.55,
                (15, 15, 15), 1, cv2.LINE_AA)


def _draw_face(frame, box, text: str, color) -> None:
    """Bounding box + label for one face (box in display coordinates)."""
    top, right, bottom, left = box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    _label(frame, text, left, top, color)


def _outline_text(frame, text: str, org, scale: float = 0.55,
                  color=(245, 245, 245)) -> None:
    """HUD text with a black outline so it stays readable on any background."""
    cv2.putText(frame, text, org, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, _FONT, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Session / record helpers
# ---------------------------------------------------------------------------
def _new_session(name: str) -> dict:
    """Create UI state for a person; check the DB ONCE for earlier check-ins
    (so people marked earlier today see 'already marked', not 'blink')."""
    premarked = database.already_marked(name)
    return {"recorded": premarked,
            "premarked": premarked,
            "welcome_until": time.time() + 3.0 if premarked else 0.0}


def _record(session: dict, name: str, now: float) -> None:
    """Write attendance to SQLite (idempotent) and flash the on-screen banner."""
    fresh = database.mark_attendance(name)
    session["recorded"] = True
    session["premarked"] = not fresh
    session["welcome_until"] = now + 3.0
    stamp = datetime.now().strftime("%H:%M:%S")
    if fresh:
        print(f"[{stamp}] ATTENDANCE RECORDED: {name}")
    else:
        print(f"[{stamp}] {name} was already marked today - nothing written.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time face recognition attendance system")
    p.add_argument("--camera", type=int, default=config.CAMERA_INDEX,
                   help="webcam index (default: %(default)s)")
    p.add_argument("--model", choices=("hog", "cnn"), default=None,
                   help="override config.MODEL (%s)" % config.MODEL)
    p.add_argument("--tolerance", type=float, default=None,
                   help="override config.TOLERANCE (%s)" % config.TOLERANCE)
    p.add_argument("--no-liveness", action="store_true",
                   help="disable the blink check - mark attendance on first sight")
    p.add_argument("--no-csv", action="store_true",
                   help="skip the automatic CSV export on exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    model = args.model or config.MODEL
    tolerance = args.tolerance if args.tolerance is not None else config.TOLERANCE

    # --- load face encodings ------------------------------------------------
    try:
        recognizer = FaceRecognizer(model=model, tolerance=tolerance)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}\n")
        return 1

    liveness = None if args.no_liveness else LivenessChecker()

    # --- open the camera ------------------------------------------------------
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("\nERROR: could not open the webcam.")
        print("  - close apps that may hold the camera (Zoom/Teams/OBS)")
        print("  - check Windows Settings > Privacy > Camera permissions")
        print("  - try another index, e.g.  python attendance_app.py --camera 1")
        return 1

    today_str = datetime.now().strftime("%Y-%m-%d")
    recorded_names = {r["name"] for r in database.get_day(today_str)}

    win_name = "Face Recognition Attendance  |  q = quit   e = export CSV"
    print("=" * 62)
    print(" Face Recognition Attendance System")
    print("=" * 62)
    print(f"  Enrolled people : {', '.join(recognizer.people)}")
    print(f"  Face encodings  : {recognizer.encoding_count}")
    print(f"  Model/tolerance : {model} / {tolerance}")
    print(f"  Liveness check  : {'ON - a blink is required' if liveness else 'OFF - marks on first sight'}")
    print(f"  Already today   : {len(recorded_names)} person(s)")
    print("  Controls        : q = quit,  e = export today's CSV")
    print("=" * 62 + "\n")

    # --- loop state -----------------------------------------------------------
    sessions: dict = {}                  # name -> session dict (see _new_session)
    fps_samples: deque = deque(maxlen=30)
    last_results: list = []              # remembered between recognition frames
    last_unknown_log = 0.0
    frame_id = 0
    scale = 1.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("WARNING: camera frame grab failed - stopping.")
            break
        if config.MIRROR_PREVIEW:
            frame = cv2.flip(frame, 1)  # selfie view; boxes stay consistent

        frame_id += 1
        t_loop = time.perf_counter()
        now = time.time()

        # downscale for the (expensive) recognition step
        h, w = frame.shape[:2]
        scale = config.PROCESS_WIDTH / w if w > config.PROCESS_WIDTH else 1.0
        if scale < 1.0:
            small = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # ---- heavy step: detect + recognise every Nth frame ----------------
        if frame_id % config.PROCESS_EVERY_N_FRAME == 0:
            last_results = recognizer.recognize(rgb_small)

        rgb_full = None  # built lazily, only when liveness needs it

        for det in last_results:
            name = det["name"]
            # scale the box from process-frame back to display coordinates
            top, right, bottom, left = det["box"]
            box = (int(top / scale), int(right / scale),
                   int(bottom / scale), int(left / scale))

            if name == "Unknown":
                _draw_face(frame, box, "Unknown", config.COLOR_UNKNOWN)
                if (config.LOG_UNKNOWN
                        and now - last_unknown_log >= config.UNKNOWN_LOG_COOLDOWN):
                    database.log_unknown()
                    last_unknown_log = now
                continue

            session = sessions.setdefault(name, _new_session(name))

            # ---- attendance decision ----------------------------------------
            if liveness is not None and not session["recorded"]:
                if rgb_full is None:
                    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                status = liveness.update(name, rgb_full, box)["status"]
                if status == LIVE:
                    _record(session, name, now)
                    recorded_names.add(name)
            elif liveness is None and not session["recorded"]:
                _record(session, name, now)          # liveness disabled
                recorded_names.add(name)

            # ---- pick label + colour ------------------------------------------
            if session["recorded"] and now < session["welcome_until"]:
                text = (f"{name} - already marked today"
                        if session["premarked"] else f"{name} - attendance recorded")
                color = config.COLOR_LIVE
            elif session["recorded"]:
                text, color = name, config.COLOR_LIVE
            else:  # still waiting for the blink
                text = (f"{name} - blink to verify" if status == ANALYZING
                        else f"{name}? no blink - photo?")
                color = (config.COLOR_ANALYZING if status == ANALYZING
                         else config.COLOR_SPOOF)

            _draw_face(frame, box, text, color)

        # ---- HUD -------------------------------------------------------------
        fps_samples.append(time.perf_counter() - t_loop)
        fps = 1.0 / (sum(fps_samples) / len(fps_samples)) if fps_samples else 0.0
        _outline_text(
            frame,
            f"FPS {fps:4.0f} | model {model} | enrolled {len(recognizer.people)}"
            f" | marked today {len(recorded_names)}"
            f" | liveness {'ON' if liveness else 'OFF'}",
            (10, 28))
        _outline_text(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, 54))

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("e"):
            print(f"CSV exported: {database.export_day_csv(today_str)}")

    # ---- clean shutdown --------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()

    if not args.no_csv:
        path = database.export_day_csv(today_str)
        rows = database.get_day(today_str)
        print(f"\nCSV exported : {path}")
        print(f"Attendance for {today_str} ({len(rows)} present):")
        for r in rows:
            print(f"  - {r['name']:<24} first seen {r['time_in']}")
    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))