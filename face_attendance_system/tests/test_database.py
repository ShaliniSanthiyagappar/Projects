"""
Unit tests for database.py - pure standard library, no dlib/camera needed.

Run from the project root:
    python tests/test_database.py
"""

import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import database  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="frs_test_"))
    db = tmp / "attendance.db"

    # 1) the once-per-day rule (the core business requirement) ---------------
    assert database.mark_attendance("Alice", "2026-01-10", "09:00:00", db_path=db) is True
    assert database.mark_attendance("Alice", "2026-01-10", "10:00:00", db_path=db) is False, \
        "duplicate entry for the same day must be rejected"
    assert database.already_marked("Alice", "2026-01-10", db_path=db)
    assert not database.already_marked("Bob", "2026-01-10", db_path=db)
    print("PASS  once-per-day rule (UNIQUE(name, date) + INSERT OR IGNORE)")

    # 2) day / range queries ---------------------------------------------------
    rows = database.get_day("2026-01-10", db_path=db)
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice" and rows[0]["time_in"] == "09:00:00"
    database.mark_attendance("Bob", "2026-01-11", "08:45:00", db_path=db)
    assert len(database.get_range("2026-01-10", "2026-01-11", db_path=db)) == 2
    print("PASS  day / range queries")

    # 3) roster + absentees from a fake dataset folder --------------------------
    ds = tmp / "dataset"
    for name in ("Alice", "Bob", "Carol"):
        (ds / name).mkdir(parents=True)
        (ds / name / "img.jpg").write_bytes(b"fake image bytes")
    assert database.enrolled_names(ds) == ["Alice", "Bob", "Carol"]
    assert database.absentees("2026-01-10", dataset_dir=ds, db_path=db) == ["Bob", "Carol"]
    print("PASS  roster (enrolled_names) + absentees")

    # 4) unknown-face audit trail ------------------------------------------------
    today = date.today().isoformat()
    database.log_unknown(db_path=db)
    database.log_unknown(db_path=db)
    assert database.unknown_count(today, db_path=db) == 2
    print("PASS  unknown sightings log")

    # 5) CSV export ----------------------------------------------------------------
    csv_path = database.export_day_csv("2026-01-10",
                                       export_dir=tmp / "exports", db_path=db)
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "Name,Date,Time In,Status"
    assert lines[1] == "Alice,2026-01-10,09:00:00,Present"
    print("PASS  CSV export ->", csv_path.name)

    # 6) trend helpers (zero-filled daily counts, per-person ranking) -----------
    database.mark_attendance("Carol", db_path=db)  # marks with real "today"
    counts = database.attendance_counts_last_days(7, db_path=db)
    assert len(counts) == 7 and counts[-1] == (today, 1)
    per_person = database.per_person_counts_last_days(30, db_path=db)
    assert per_person and per_person[0] == ("Carol", 1)
    print("PASS  trend helpers")

    print("\nAll database tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())