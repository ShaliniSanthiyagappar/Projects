"""
dashboard.py -- Streamlit analytics dashboard for the attendance system.

Run from the project folder (keep the venv active):

    streamlit run dashboard.py

Three pages (radio switch in the sidebar):
  * Today      - KPI cards + the day's check-in table + one-click CSV export
  * Analytics  - 14-day head-count trend, per-person ranking (30 d),
                 check-in hour distribution (30 d)
  * Absentees  - who is missing today vs the dataset/ roster, unknown-face
                 sightings, full-history CSV export

Data source: the same data/attendance.db that attendance_app.py writes to,
so the dashboard always reflects what the camera recorded. Queries are
cached for 15 s to keep the UI snappy; use "Refresh now" after a session.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

import config
import database

st.set_page_config(page_title="Face Attendance Dashboard", layout="wide")

TEAL = "#2a9d8f"      # primary chart colour
DARK = "#264653"      # secondary chart colour


# ---------------------------------------------------------------------------
# Cached data loaders (15 s TTL keeps charts responsive; Refresh clears them)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=15)
def load_day(date_str: str) -> pd.DataFrame:
    conn = database.connect()
    try:
        return pd.read_sql(
            "SELECT name, date, time_in, status FROM attendance "
            "WHERE date = ? ORDER BY time_in",
            conn, params=(date_str,))
    finally:
        conn.close()


@st.cache_data(ttl=15)
def load_range(start: str, end: str) -> pd.DataFrame:
    conn = database.connect()
    try:
        return pd.read_sql(
            "SELECT name, date, time_in FROM attendance "
            "WHERE date BETWEEN ? AND ? ORDER BY date, time_in",
            conn, params=(start, end))
    finally:
        conn.close()


def _late_count(df: pd.DataFrame) -> int:
    """Rows whose check-in time is after config.LATE_AFTER."""
    if df.empty:
        return 0
    try:
        cutoff = datetime.strptime(config.LATE_AFTER, "%H:%M").time()
        return int(sum(datetime.strptime(t, "%H:%M:%S").time() > cutoff
                       for t in df["time_in"]))
    except ValueError:
        return 0


def _full_history_csv() -> bytes:
    conn = database.connect()
    try:
        df = pd.read_sql(
            "SELECT name, date, time_in, status FROM attendance "
            "ORDER BY date, time_in", conn)
    finally:
        conn.close()
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_today(day: str) -> None:
    st.subheader(f"Attendance - {day}")
    df = load_day(day)
    enrolled = database.enrolled_names()
    present = set(df["name"]) if not df.empty else set()
    absent = [n for n in enrolled if n not in present]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Enrolled", len(enrolled))
    k2.metric("Present", len(present))
    k3.metric("Absent", len(absent))
    k4.metric("Late (after %s)" % config.LATE_AFTER, _late_count(df))
    k5.metric("Unknown sightings", database.unknown_count(day))

    if df.empty:
        st.info("No attendance records for this day yet. Run "
                "`python attendance_app.py` and let people look into the camera.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Export this day as CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       file_name=f"attendance_{day}.csv", mime="text/csv")
    if absent:
        st.caption("Absent: " + ", ".join(absent))


def page_analytics() -> None:
    st.subheader("Analytics")
    today = date_cls.today()

    counts = database.attendance_counts_last_days(14)
    trend = pd.DataFrame(counts, columns=["Date", "Present"])
    fig_trend = px.bar(trend, x="Date", y="Present",
                       color_discrete_sequence=[TEAL],
                       title="Daily head-count (last 14 days)")
    fig_trend.update_layout(height=340, margin=dict(t=42, b=0))
    st.plotly_chart(fig_trend, use_container_width=True)

    left, right = st.columns(2)

    with left:
        per_person = pd.DataFrame(database.per_person_counts_last_days(30),
                                  columns=["Person", "Days present"])
        if per_person.empty:
            st.info("No data in the last 30 days yet.")
        else:
            fig_p = px.bar(per_person.sort_values("Days present"),
                           x="Days present", y="Person", orientation="h",
                           color_discrete_sequence=[TEAL],
                           title="Days present per person (30 days)")
            fig_p.update_layout(height=400, margin=dict(t=42))
            st.plotly_chart(fig_p, use_container_width=True)

    with right:
        start30 = (today - timedelta(days=29)).isoformat()
        hist = load_range(start30, today.isoformat())
        if hist.empty:
            st.info("No check-ins in the last 30 days yet.")
        else:
            hist["hour"] = pd.to_datetime(hist["time_in"],
                                          format="%H:%M:%S").dt.hour
            fig_h = px.histogram(hist, x="hour", nbins=24,
                                 color_discrete_sequence=[DARK],
                                 title="Check-in hour distribution (30 days)")
            fig_h.update_layout(height=400, margin=dict(t=42),
                                xaxis_title="Hour of day",
                                yaxis_title="Check-ins")
            st.plotly_chart(fig_h, use_container_width=True)


def page_absentees(day: str) -> None:
    st.subheader(f"Absentees - {day}")
    absent = database.absentees(day)

    c1, c2 = st.columns(2)
    c1.metric("Absent", len(absent))
    c2.metric("Unknown sightings", database.unknown_count(day))

    if absent:
        st.warning("Not marked yet: " + ", ".join(absent))
        st.caption("New person? Add a folder dataset/<Name>/ with 3-10 clear "
                   "photos, then re-run `python dataset_builder.py`.")
    else:
        st.success("Everyone enrolled is marked present for this day.")

    st.divider()
    st.download_button("Export full history as CSV", _full_history_csv(),
                       file_name="attendance_full_history.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("Face Recognition Attendance")

    with st.sidebar:
        st.header("Controls")
        chosen = st.date_input("Date", value=date_cls.today())
        page = st.radio("Page", ("Today", "Analytics", "Absentees"))
        if st.button("Refresh now"):
            st.cache_data.clear()
        st.caption("Data is cached for 15 s. Click *Refresh now* after a "
                   "recognition session to see the latest rows.")

    day = chosen.isoformat()
    if page == "Today":
        page_today(day)
    elif page == "Analytics":
        page_analytics()
    else:
        page_absentees(day)


if __name__ == "__main__":
    main()