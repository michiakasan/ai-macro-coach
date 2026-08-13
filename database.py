"""
SQLite persistence for MacroTrack.

Everything here is plain `sqlite3` from the standard library -- no ORM, so the SQL
that produces the running totals is visible and easy to explain.

Design note for the architecture story: this module is the *only* place macro totals
are produced. They come from `SUM(...)` over stored rows, i.e. from the database, not
from the language model. chains.py never imports this module; app.py wires the two
together. That keeps the boundary clean:

    LLM  -> estimates one meal        (chains.estimate_meal)
    SQL  -> sums meals into totals    (this module)
    Python -> goal - consumed = gap   (models.MacroTotals.remaining_from)
    LLM  -> turns that gap into ideas (chains.recommend_meals)
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

from models import MEAL_SLOTS, Goals, MacroTotals

# The DB file lives next to the code by default; override with MACROTRACK_DB in .env
# (handy for tests, or for keeping the data outside the repo).
DB_PATH = os.getenv("MACROTRACK_DB") or str(Path(__file__).parent / "macrotrack.db")


SCHEMA = """
-- Single-row table: there is one user, so the goals row is pinned to id = 1.
CREATE TABLE IF NOT EXISTS goals (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    daily_calories    REAL NOT NULL,
    daily_protein_g   REAL NOT NULL,
    daily_carbs_g     REAL NOT NULL,
    daily_fat_g       REAL NOT NULL,
    -- Weekly targets are optional; NULL means "just use 7 x daily".
    weekly_calories   REAL,
    weekly_protein_g  REAL,
    weekly_carbs_g    REAL,
    weekly_fat_g      REAL,
    updated_at        TEXT NOT NULL
);

-- One row per logged meal. `log_date` is a plain YYYY-MM-DD string so that
-- day/week grouping is a straight string comparison in SQL.
CREATE TABLE IF NOT EXISTS meals (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    log_date   TEXT    NOT NULL,
    logged_at  TEXT    NOT NULL,
    meal_name  TEXT    NOT NULL,
    -- breakfast / lunch / dinner / snack -- the columns of the eating-pattern table.
    meal_slot  TEXT    NOT NULL DEFAULT 'snack',
    calories   REAL    NOT NULL DEFAULT 0,
    protein_g  REAL    NOT NULL DEFAULT 0,
    carbs_g    REAL    NOT NULL DEFAULT 0,
    fat_g      REAL    NOT NULL DEFAULT 0,
    -- 1 = numbers came straight from the LLM, 0 = the user hand-corrected them.
    estimated  INTEGER NOT NULL DEFAULT 1,
    assumptions TEXT   NOT NULL DEFAULT '',
    raw_input  TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_meals_log_date ON meals (log_date);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a connection with dict-like rows, committing on clean exit."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist, then apply migrations. Safe on every start."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive migrations for databases created by an earlier version.

    `CREATE TABLE IF NOT EXISTS` silently does nothing when the table already exists,
    so a new column has to be added explicitly or existing logs would break on read.
    """
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(meals)")}

    if "meal_slot" not in columns:
        # Older rows have no slot recorded; 'snack' is the honest neutral default and
        # shows up in the pattern table as an uncategorised entry the user can re-log.
        conn.execute("ALTER TABLE meals ADD COLUMN meal_slot TEXT NOT NULL DEFAULT 'snack'")


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


def save_goals(goals: Goals) -> None:
    """Insert or replace the single goals row."""
    weekly = goals.weekly
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO goals (
                id, daily_calories, daily_protein_g, daily_carbs_g, daily_fat_g,
                weekly_calories, weekly_protein_g, weekly_carbs_g, weekly_fat_g, updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                daily_calories   = excluded.daily_calories,
                daily_protein_g  = excluded.daily_protein_g,
                daily_carbs_g    = excluded.daily_carbs_g,
                daily_fat_g      = excluded.daily_fat_g,
                weekly_calories  = excluded.weekly_calories,
                weekly_protein_g = excluded.weekly_protein_g,
                weekly_carbs_g   = excluded.weekly_carbs_g,
                weekly_fat_g     = excluded.weekly_fat_g,
                updated_at       = excluded.updated_at
            """,
            (
                goals.daily.calories,
                goals.daily.protein_g,
                goals.daily.carbs_g,
                goals.daily.fat_g,
                weekly.calories if weekly else None,
                weekly.protein_g if weekly else None,
                weekly.carbs_g if weekly else None,
                weekly.fat_g if weekly else None,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def get_goals() -> Goals | None:
    """Return the saved goals, or None if the user hasn't set any yet."""
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM goals WHERE id = 1").fetchone()

    if row is None:
        return None

    daily = MacroTotals(
        calories=row["daily_calories"],
        protein_g=row["daily_protein_g"],
        carbs_g=row["daily_carbs_g"],
        fat_g=row["daily_fat_g"],
    )
    # Weekly is only "set" if all four columns are populated.
    weekly_cols = [row["weekly_calories"], row["weekly_protein_g"], row["weekly_carbs_g"], row["weekly_fat_g"]]
    weekly = (
        MacroTotals(
            calories=weekly_cols[0],
            protein_g=weekly_cols[1],
            carbs_g=weekly_cols[2],
            fat_g=weekly_cols[3],
        )
        if all(c is not None for c in weekly_cols)
        else None
    )
    return Goals(daily=daily, weekly=weekly)


# ---------------------------------------------------------------------------
# Meals
# ---------------------------------------------------------------------------


def add_meal(
    *,
    log_date: date,
    meal_name: str,
    macros: MacroTotals,
    meal_slot: str = "snack",
    estimated: bool = True,
    assumptions: str = "",
    raw_input: str = "",
) -> int:
    """Persist one meal entry and return its row id."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO meals (
                log_date, logged_at, meal_name, meal_slot,
                calories, protein_g, carbs_g, fat_g,
                estimated, assumptions, raw_input
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_date.isoformat(),
                datetime.now().isoformat(timespec="seconds"),
                meal_name,
                meal_slot,
                macros.calories,
                macros.protein_g,
                macros.carbs_g,
                macros.fat_g,
                1 if estimated else 0,
                assumptions,
                raw_input,
            ),
        )
        return int(cur.lastrowid)


def delete_meal(meal_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM meals WHERE id = ?", (meal_id,))


def get_meals_for_date(log_date: date) -> list[sqlite3.Row]:
    """All entries for one day, in slot order (breakfast -> snack), then oldest first."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT * FROM meals
            WHERE log_date = ?
            ORDER BY CASE meal_slot
                         WHEN 'breakfast' THEN 0
                         WHEN 'lunch'     THEN 1
                         WHEN 'dinner'    THEN 2
                         ELSE 3
                     END,
                     logged_at ASC
            """,
            (log_date.isoformat(),),
        ).fetchall()


def _row_to_totals(row: sqlite3.Row | None) -> MacroTotals:
    """SUM() returns NULL for an empty set, so coalesce to 0.0."""
    if row is None:
        return MacroTotals()
    return MacroTotals(
        calories=row["calories"] or 0.0,
        protein_g=row["protein_g"] or 0.0,
        carbs_g=row["carbs_g"] or 0.0,
        fat_g=row["fat_g"] or 0.0,
    )


def get_totals_for_date(log_date: date) -> MacroTotals:
    """Consumed totals for a single day, summed in SQL."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT SUM(calories)  AS calories,
                   SUM(protein_g) AS protein_g,
                   SUM(carbs_g)   AS carbs_g,
                   SUM(fat_g)     AS fat_g
            FROM meals
            WHERE log_date = ?
            """,
            (log_date.isoformat(),),
        ).fetchone()
    return _row_to_totals(row)


def get_totals_between(start: date, end: date) -> MacroTotals:
    """Consumed totals across an inclusive date range (used for the weekly view)."""
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT SUM(calories)  AS calories,
                   SUM(protein_g) AS protein_g,
                   SUM(carbs_g)   AS carbs_g,
                   SUM(fat_g)     AS fat_g
            FROM meals
            WHERE log_date BETWEEN ? AND ?
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchone()
    return _row_to_totals(row)


def get_daily_totals_range(start: date, end: date) -> dict[date, MacroTotals]:
    """Per-day totals for an inclusive range, with zero-filled days.

    Days with no meals still appear (as zeros) so the weekly chart has a bar for
    every day rather than silently skipping gaps.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT log_date,
                   SUM(calories)  AS calories,
                   SUM(protein_g) AS protein_g,
                   SUM(carbs_g)   AS carbs_g,
                   SUM(fat_g)     AS fat_g
            FROM meals
            WHERE log_date BETWEEN ? AND ?
            GROUP BY log_date
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()

    by_date = {date.fromisoformat(r["log_date"]): _row_to_totals(r) for r in rows}

    out: dict[date, MacroTotals] = {}
    day = start
    while day <= end:
        out[day] = by_date.get(day, MacroTotals())
        day += timedelta(days=1)
    return out


def get_meals_by_slot_range(start: date, end: date) -> dict[date, dict[str, list[sqlite3.Row]]]:
    """Meals grouped as {day: {slot: [rows]}} for the eating-pattern table.

    Every day in the range and every slot is present (possibly empty), so the table
    has a complete grid and a skipped breakfast is visibly blank rather than missing.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM meals
            WHERE log_date BETWEEN ? AND ?
            ORDER BY log_date ASC, logged_at ASC
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()

    grid: dict[date, dict[str, list[sqlite3.Row]]] = {}
    day = start
    while day <= end:
        grid[day] = {slot: [] for slot in MEAL_SLOTS}
        day += timedelta(days=1)

    for row in rows:
        slot = row["meal_slot"] if row["meal_slot"] in MEAL_SLOTS else "snack"
        grid[date.fromisoformat(row["log_date"])][slot].append(row)
    return grid


def get_slot_stats(start: date, end: date) -> dict[str, dict[str, float]]:
    """Per-slot habits over a range: how often it's eaten and its average macros.

    All of this is SQL aggregation plus one division in Python -- the model is given
    the finished figures, never the rows to average itself.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT meal_slot,
                   COUNT(DISTINCT log_date) AS days_logged,
                   SUM(calories)  AS calories,
                   SUM(protein_g) AS protein_g,
                   SUM(carbs_g)   AS carbs_g,
                   SUM(fat_g)     AS fat_g
            FROM meals
            WHERE log_date BETWEEN ? AND ?
            GROUP BY meal_slot
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()

    by_slot = {r["meal_slot"]: r for r in rows}
    stats: dict[str, dict[str, float]] = {}

    for slot in MEAL_SLOTS:
        row = by_slot.get(slot)
        days = row["days_logged"] if row else 0
        # Average over the days this slot was actually eaten, not over the whole
        # window -- otherwise skipping breakfast looks like "a small breakfast".
        divisor = days or 1
        stats[slot] = {
            "days_logged": days,
            "avg_calories": (row["calories"] or 0) / divisor if row else 0.0,
            "avg_protein_g": (row["protein_g"] or 0) / divisor if row else 0.0,
            "avg_carbs_g": (row["carbs_g"] or 0) / divisor if row else 0.0,
            "avg_fat_g": (row["fat_g"] or 0) / divisor if row else 0.0,
        }
    return stats


def last_seven_days(today: date) -> tuple[date, date]:
    """Inclusive (start, end) covering today and the six days before it."""
    return today - timedelta(days=6), today


def last_n_days(today: date, n: int) -> tuple[date, date]:
    """Inclusive (start, end) covering today and the n-1 days before it."""
    return today - timedelta(days=n - 1), today
