"""
database.py -- SQLite storage layer for the attendance system.

Why the once-per-day rule is bullet-proof
-----------------------------------------
The `attendance` table carries a UNIQUE(name, date) constraint and rows are
written with `INSERT OR IGNORE`. The database itself rejects duplicates, so
even two app instances running at the same time cannot mark a person twice.

This module intentionally uses only the Python standard library, so it can be
unit-tested on machines without OpenCV / face_recognition installed
(see tests/test_database.py).
"""

from __future__ import annotations

import csv
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

from config import DATASET_DIR, DB_FILE, EXPORT_DIR, IMAGE_EXTS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attendance (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT NOT NULL,
    date    TEXT NOT NULL,               -- YYYY-MM-DD (local time)
    time_in TEXT NOT NULL,               -- HH:MM:SS (local time)
    status  TEXT NOT NULL DEFAULT 'Present',
    UNIQUE (name, date)                  -- <- enforces "only once per day"
);
CREATE TABLE IF NOT EXISTS unknown_sightings (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,                  -- YYYY-MM-DD
    time TEXT NOT NULL                   -- HH:MM:SS
);
"""


def connect(db_path: "Path | str" = DB_FILE) -> sqlite3.Connection:
    """Open (and initialise) the database; returns a sqlite3 connection."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Attendance rows
# ---------------------------------------------------------------------------
def mark_attendance(name: str,
                    date_str: Optional[str] = None,
                    time_str: Optional[str] = None,
                    status: str = "Present",
                    db_path: "Path | str" = DB_FILE) -> bool:
    """Record one person's check-in.

    Returns True when a NEW row was inserted, False when the person was
    already marked on that date (duplicate blocked by the DB itself).
    """
    now = datetime.now()
    date_str = date_str or now.strftime("%Y-%m-%d")
    time_str = time_str or now.strftime("%H:%M:%S")
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "INSERT OR IGNORE INTO attendance (name, date, time_in, status) "
            "VALUES (?, ?, ?, ?)",
            (name, date_str, time_str, status),
        )
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def already_marked(name: str,
                   date_str: Optional[str] = None,
                   db_path: "Path | str" = DB_FILE) -> bool:
    """True if `name` already has a row for `date_str` (default: today)."""
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM attendance WHERE name = ? AND date = ?",
            (name, date_str),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def get_day(date_str: str, db_path: "Path | str" = DB_FILE) -> List[sqlite3.Row]:
    """All attendance rows for one day, ordered by check-in time."""
    conn = connect(db_path)
    try:
        return conn.execute(
            "SELECT name, date, time_in, status FROM attendance "
            "WHERE date = ? ORDER BY time_in",
            (date_str,),
        ).fetchall()
    finally:
        conn.close()


def get_range(start_date: str, end_date: str,
              db_path: "Path | str" = DB_FILE) -> List[sqlite3.Row]:
    """All attendance rows between two dates (inclusive)."""
    conn = connect(db_path)
    try:
        return conn.execute(
            "SELECT name, date, time_in, status FROM attendance "
            "WHERE date BETWEEN ? AND ? ORDER BY date, time_in",
            (start_date, end_date),
        ).fetchall()
    finally:
        conn.close()


def all_rows(db_path: "Path | str" = DB_FILE) -> List[sqlite3.Row]:
    """Every attendance row ever recorded (for full-history exports)."""
    conn = connect(db_path)
    try:
        return conn.execute(
            "SELECT name, date, time_in, status FROM attendance ORDER BY date, time_in"
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Unknown-face audit trail
# ---------------------------------------------------------------------------
def log_unknown(db_path: "Path | str" = DB_FILE) -> None:
    """Append one timestamped row for an unregistered face sighting."""
    now = datetime.now()
    conn = connect(db_path)
    try:
        conn.execute(
            "INSERT INTO unknown_sightings (date, time) VALUES (?, ?)",
            (now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")),
        )
        conn.commit()
    finally:
        conn.close()


def unknown_count(date_str: str, db_path: "Path | str" = DB_FILE) -> int:
    """How many unknown-face sightings were logged on `date_str`."""
    conn = connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM unknown_sightings WHERE date = ?",
            (date_str,),
        ).fetchone()
        return int(row["n"]) if row else 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Roster helpers (derived from the dataset/ folder)
# ---------------------------------------------------------------------------
def enrolled_names(dataset_dir: "Path | str" = DATASET_DIR) -> List[str]:
    """Names of all people who have at least one image in dataset/<Name>/."""
    root = Path(dataset_dir)
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and any(f.suffix.lower() in IMAGE_EXTS for f in p.iterdir())
    )


def absentees(date_str: str,
              dataset_dir: "Path | str" = DATASET_DIR,
              db_path: "Path | str" = DB_FILE) -> List[str]:
    """Enrolled people with NO attendance row on `date_str`."""
    present = {r["name"] for r in get_day(date_str, db_path)}
    return [n for n in enrolled_names(dataset_dir) if n not in present]


# ---------------------------------------------------------------------------
# CSV export + analytics helpers
# ---------------------------------------------------------------------------
def export_day_csv(date_str: Optional[str] = None,
                   export_dir: "Path | str" = EXPORT_DIR,
                   db_path: "Path | str" = DB_FILE) -> Path:
    """Write one day's attendance to exports/attendance_<date>.csv.

    Returns the path of the written file. Called automatically when the
    attendance app exits (and available from the dashboard).
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"attendance_{date_str}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time In", "Status"])
        for row in get_day(date_str, db_path):
            writer.writerow([row["name"], row["date"], row["time_in"], row["status"]])
    return path


def attendance_counts_last_days(n: int = 14,
                                db_path: "Path | str" = DB_FILE) -> List[Tuple[str, int]]:
    """(date, head-count) for the last n days, zero-filled for missing days."""
    today = datetime.now().date()
    start = (today - timedelta(days=n - 1)).isoformat()
    counts = Counter(r["date"] for r in get_range(start, today.isoformat(), db_path))
    return [
        ((today - timedelta(days=i)).isoformat(), counts.get((today - timedelta(days=i)).isoformat(), 0))
        for i in range(n - 1, -1, -1)
    ]


def per_person_counts_last_days(n: int = 30,
                                db_path: "Path | str" = DB_FILE) -> List[Tuple[str, int]]:
    """(name, days present) ranked by attendance over the last n days."""
    today = datetime.now().date()
    start = (today - timedelta(days=n - 1)).isoformat()
    counts = Counter(r["name"] for r in get_range(start, today.isoformat(), db_path))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))