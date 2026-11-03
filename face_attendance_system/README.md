Face Recognition Attendance System
Real-time, multi-face attendance for a classroom or office using a singlewebcam. Built with Python + OpenCV + face_recognition (dlib) + SQLite +Streamlit.

Detects multiple faces simultaneously and matches them against yourenrolled roster (dataset/<PersonName>/ photos).
Marks Name, Date, Time in SQLite — and the database itself guarantees aperson is marked only once per day (UNIQUE(name, date) + INSERT OR IGNORE).
Draws a coloured bounding box + name on the live video, colour-coded byliveness state.
Liveness detection (flagship feature 1): a blink is required beforeattendance is written, so holding up a photo does not work.
Streamlit dashboard (flagship feature 2): KPIs, 14-day trend, per-personranking, check-in hour distribution, absentees, one-click CSV export.
Unknown-face audit trail: unregistered faces get a red box and athrottled timestamped row in the database.
Automatic CSV export to exports/attendance_<date>.csv every session.
