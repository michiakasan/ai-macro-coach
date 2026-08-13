"""
MacroTrack -- Streamlit frontend.

This file is the wiring layer. It owns no nutrition logic of its own:

    chains.py    -> asks the LLM to estimate one meal, and later to suggest food
    database.py  -> stores meals and SUMs them into consumed totals
    models.py    -> holds the goal/consumed/remaining arithmetic
    app.py (here)-> reads totals from the DB, computes the gap in Python, renders it,
                    and hands that finished gap to the recommendation chain

The order of operations in the Dashboard tab is the thing worth pointing at:
totals come from SQL, the gap is subtraction in Python, and only *then* does a second
LLM call happen -- with the numbers already settled.
"""

from __future__ import annotations

import html
import sqlite3
from datetime import date, datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

import database as db
from chains import (
    MissingAPIKeyError,
    current_meal_slot,
    estimate_meal,
    format_macros,
    recommend_meals,
    time_of_day_label,
)
from models import (
    MACRO_FIELDS,
    MACRO_LABELS,
    MEAL_SLOTS,
    SLOT_LABELS,
    Goals,
    MacroTotals,
    MealEstimate,
)

# How many days the breakfast/lunch/dinner pattern table covers.
PATTERN_DAYS = 6

# Load OPENAI_API_KEY (and any overrides) from .env before anything builds a chain.
load_dotenv()

st.set_page_config(page_title="MacroTrack", page_icon="🥗", layout="wide")

# Create tables on every start; it's a no-op once they exist.
db.init_db()


# ---------------------------------------------------------------------------
# Small rendering helpers
# ---------------------------------------------------------------------------


def macro_progress(consumed: MacroTotals, goal: MacroTotals) -> None:
    """Render one metric + progress bar per macro: 'Protein 85g / 140g goal'."""
    percents = consumed.percent_of(goal)
    columns = st.columns(len(MACRO_FIELDS))

    for column, field in zip(columns, MACRO_FIELDS):
        label, unit = MACRO_LABELS[field]
        have = getattr(consumed, field)
        want = getattr(goal, field)
        left = want - have

        with column:
            st.metric(
                label=f"{label} ({unit})",
                value=f"{have:,.0f} / {want:,.0f}",
                delta=(f"{left:,.0f} to go" if left > 0 else f"{abs(left):,.0f} over"),
                delta_color="normal" if left > 0 else "inverse",
            )
            # st.progress needs 0..1, so cap the bar while the metric above still
            # shows the true overshoot.
            st.progress(min(percents[field], 1.0))


def totals_frame(per_day: dict[date, MacroTotals]) -> pd.DataFrame:
    """Turn per-day totals into a tidy DataFrame for the weekly table/chart."""
    return pd.DataFrame(
        [
            {
                "Date": day.isoformat(),
                "Day": day.strftime("%a"),
                "Calories": round(totals.calories),
                "Protein (g)": round(totals.protein_g),
                "Carbs (g)": round(totals.carbs_g),
                "Fat (g)": round(totals.fat_g),
            }
            for day, totals in per_day.items()
        ]
    )


def _day_calories(slots: dict[str, list[sqlite3.Row]]) -> float:
    """Total calories for one day of the pattern grid (Python, not the LLM)."""
    return sum(row["calories"] for rows in slots.values() for row in rows)


def pattern_markdown(grid: dict[date, dict[str, list[sqlite3.Row]]], today: date) -> str:
    """Render the day x slot grid as a markdown table, most recent day first.

    Built by hand rather than with st.dataframe because a cell can hold several meals
    and needs to wrap -- a dataframe would truncate them to one clipped line.
    """

    def cell(rows: list[sqlite3.Row]) -> str:
        if not rows:
            return "—"
        # Escape HTML and the markdown cell separator: meal names are free text.
        return "<br>".join(
            f"{html.escape(r['meal_name']).replace('|', '&#124;')} "
            f"<sub>{r['calories']:,.0f} kcal · {r['protein_g']:.0f}g P</sub>"
            for r in rows
        )

    header = "| Day | " + " | ".join(SLOT_LABELS[s] for s in MEAL_SLOTS) + " | Total |"
    divider = "|---" * (len(MEAL_SLOTS) + 2) + "|"
    lines = [header, divider]

    for day in sorted(grid, reverse=True):
        slots = grid[day]
        name = day.strftime("%a %d %b")
        if day == today:
            name = f"**{name}** (today)"
        total = _day_calories(slots)
        cells = " | ".join(cell(slots[s]) for s in MEAL_SLOTS)
        lines.append(f"| {name} | {cells} | {total:,.0f} kcal |")

    return "\n".join(lines)


def pattern_for_prompt(grid: dict[date, dict[str, list[sqlite3.Row]]], today: date) -> str:
    """Plain-text version of the same grid, for the recommendation prompt.

    Same data the user sees on screen, so a suggestion that references "your usual
    breakfast" is grounded in rows they can actually point at.
    """
    blocks = []
    for day in sorted(grid, reverse=True):
        slots = grid[day]
        if not any(slots.values()):
            blocks.append(f"{day.strftime('%a %d %b')}: nothing logged")
            continue

        label = f"{day.strftime('%a %d %b')}{' (today)' if day == today else ''}"
        lines = [f"{label} -- {_day_calories(slots):,.0f} kcal total"]
        for slot in MEAL_SLOTS:
            rows = slots[slot]
            if not rows:
                lines.append(f"  {slot}: skipped / not logged")
                continue
            items = "; ".join(
                f"{r['meal_name']} ({r['calories']:,.0f} kcal, {r['protein_g']:.0f}g P, "
                f"{r['carbs_g']:.0f}g C, {r['fat_g']:.0f}g F)"
                for r in rows
            )
            lines.append(f"  {slot}: {items}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def slot_habits_for_prompt(stats: dict[str, dict[str, float]], window_days: int) -> str:
    """One line per slot: frequency and average macros, both computed in SQL/Python."""
    lines = []
    for slot in MEAL_SLOTS:
        s = stats[slot]
        days = int(s["days_logged"])
        if days == 0:
            lines.append(f"{SLOT_LABELS[slot]}: never logged in the last {window_days} days")
            continue
        lines.append(
            f"{SLOT_LABELS[slot]}: eaten on {days} of the last {window_days} days, "
            f"averaging {s['avg_calories']:,.0f} kcal | {s['avg_protein_g']:.0f}g protein | "
            f"{s['avg_carbs_g']:.0f}g carbs | {s['avg_fat_g']:.0f}g fat on those days"
        )
    return "\n".join(lines)


def weekly_context_line(per_day: dict[date, MacroTotals], daily_goal: MacroTotals) -> str:
    """A one-line summary of the week, computed in Python, for the recommendation prompt.

    Gives the model a sense of the trend ("protein has been short all week") without
    ever handing it raw rows to add up.
    """
    logged_days = [d for d, t in per_day.items() if t.calories > 0]
    if not logged_days:
        return "no meals logged in the last 7 days"

    week_total = MacroTotals()
    for totals in per_day.values():
        week_total = week_total + totals

    avg = week_total.scaled(1 / len(logged_days))
    return (
        f"{len(logged_days)} of the last 7 days have logged meals; "
        f"daily average {format_macros(avg)} against a daily goal of {format_macros(daily_goal)}"
    )


def require_goals() -> Goals | None:
    """Fetch goals, or show a nudge toward the Set Goals tab."""
    goals = db.get_goals()
    if goals is None:
        st.info("No goals set yet. Open the **Set Goals** tab to get started.")
    return goals


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🥗 MacroTrack")
st.caption("Log meals in plain language, track macros against your goals, get gap-aware suggestions.")

log_tab, dashboard_tab, goals_tab = st.tabs(["Log Meal", "Dashboard", "Set Goals"])


# ---------------------------------------------------------------------------
# TAB 1 -- Log a meal
# ---------------------------------------------------------------------------

with log_tab:
    st.subheader("What did you eat?")

    with st.form("meal_form"):
        description = st.text_area(
            "Describe the meal in your own words",
            placeholder="grilled chicken breast and rice for lunch, about 200g chicken",
            height=100,
        )
        log_date = st.date_input("Date", value=date.today())
        submitted = st.form_submit_button("Estimate macros", type="primary")

    if submitted:
        if not description.strip():
            st.warning("Type what you ate first.")
        else:
            try:
                with st.spinner("Estimating macros..."):
                    # LLM CALL #1: free text -> structured MealEstimate.
                    estimate = estimate_meal(description)
                # Park it in session state; nothing is saved until the user confirms.
                st.session_state["pending_estimate"] = estimate
                st.session_state["pending_raw"] = description
                st.session_state["pending_date"] = log_date
            except MissingAPIKeyError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001 -- surface API/parse failures in the UI
                st.error(f"Could not estimate this meal: {exc}")

    # --- confirm / adjust step: the user always has the final say ---
    pending: MealEstimate | None = st.session_state.get("pending_estimate")
    if pending is not None:
        st.divider()
        st.markdown("### Check the estimate before saving")
        st.caption(
            "These are LLM estimates, not lab values. Adjust anything that looks off — "
            "your edits are what get stored."
        )

        if pending.assumptions:
            st.info(f"**Assumptions the model made:** {pending.assumptions}")

        with st.form("confirm_form"):
            name_col, slot_col = st.columns([3, 1])
            meal_name = name_col.text_input("Meal name", value=pending.meal_name)
            # Pre-selected with the slot the model inferred; the user can correct it.
            meal_slot = slot_col.selectbox(
                "Meal",
                options=list(MEAL_SLOTS),
                index=list(MEAL_SLOTS).index(pending.meal_slot),
                format_func=lambda s: SLOT_LABELS[s],
            )
            c1, c2, c3, c4 = st.columns(4)
            calories = c1.number_input("Calories", min_value=0.0, value=float(pending.calories), step=10.0)
            protein = c2.number_input("Protein (g)", min_value=0.0, value=float(pending.protein_g), step=1.0)
            carbs = c3.number_input("Carbs (g)", min_value=0.0, value=float(pending.carbs_g), step=1.0)
            fat = c4.number_input("Fat (g)", min_value=0.0, value=float(pending.fat_g), step=1.0)

            save_col, discard_col = st.columns(2)
            save = save_col.form_submit_button("Save to log", type="primary")
            discard = discard_col.form_submit_button("Discard")

        if save:
            edited = MacroTotals(calories=calories, protein_g=protein, carbs_g=carbs, fat_g=fat)
            # If the user changed a number, it's no longer a pure LLM estimate --
            # record that distinction rather than labelling hand-corrected data "estimated".
            still_estimated = edited == pending.totals()

            db.add_meal(
                log_date=st.session_state.get("pending_date", date.today()),
                meal_name=meal_name,
                macros=edited,
                meal_slot=meal_slot,
                estimated=still_estimated,
                assumptions=pending.assumptions,
                raw_input=st.session_state.get("pending_raw", ""),
            )
            for key in ("pending_estimate", "pending_raw", "pending_date"):
                st.session_state.pop(key, None)
            st.success(f"Saved: {meal_name}")
            st.rerun()

        if discard:
            for key in ("pending_estimate", "pending_raw", "pending_date"):
                st.session_state.pop(key, None)
            st.rerun()

    # --- today's entries ---
    st.divider()
    view_date = st.date_input("Show entries for", value=date.today(), key="log_view_date")
    meals = db.get_meals_for_date(view_date)

    if not meals:
        st.caption("Nothing logged for this day yet.")
    else:
        for meal in meals:
            with st.container(border=True):
                text_col, delete_col = st.columns([6, 1])
                with text_col:
                    flag = "~ estimated" if meal["estimated"] else "✎ user-adjusted"
                    macros = MacroTotals(
                        calories=meal["calories"],
                        protein_g=meal["protein_g"],
                        carbs_g=meal["carbs_g"],
                        fat_g=meal["fat_g"],
                    )
                    slot_label = SLOT_LABELS.get(meal["meal_slot"], "Snacks")
                    st.markdown(f"**{slot_label} — {meal['meal_name']}**  \n{format_macros(macros)}")
                    st.caption(f"{meal['logged_at'][11:16]} · {flag}")
                    if meal["assumptions"]:
                        st.caption(f"Assumptions: {meal['assumptions']}")
                if delete_col.button("Delete", key=f"del_{meal['id']}"):
                    db.delete_meal(meal["id"])
                    st.rerun()


# ---------------------------------------------------------------------------
# TAB 2 -- Dashboard (totals, gaps, recommendations)
# ---------------------------------------------------------------------------

with dashboard_tab:
    goals = require_goals()

    if goals is not None:
        target_date = st.date_input("Day", value=date.today(), key="dash_date")

        # ---- STEP 1: totals come from SQL, not the LLM ----
        consumed = db.get_totals_for_date(target_date)
        daily_goal = goals.daily

        # ---- STEP 2: the gap is plain Python subtraction ----
        remaining = consumed.remaining_from(daily_goal)
        behind = consumed.furthest_behind(daily_goal)

        st.subheader(f"Today vs. goal — {target_date.strftime('%A %d %b')}")
        macro_progress(consumed, daily_goal)

        if behind is None:
            st.success("All daily targets met. 🎉")
        else:
            label, unit = MACRO_LABELS[behind]
            gap_text = f"{getattr(remaining, behind):,.0f}" + (" kcal" if behind == "calories" else unit)
            st.warning(
                f"**Furthest behind: {label}** — {gap_text} still to go "
                f"({consumed.percent_of(daily_goal)[behind] * 100:.0f}% of goal)."
            )

        # ---- Weekly view ----
        st.divider()
        st.subheader("Last 7 days")

        start, end = db.last_seven_days(target_date)
        per_day = db.get_daily_totals_range(start, end)
        frame = totals_frame(per_day)

        chart_col, table_col = st.columns([1, 1])
        with chart_col:
            metric_choice = st.selectbox(
                "Chart metric", ["Calories", "Protein (g)", "Carbs (g)", "Fat (g)"], index=0
            )
            st.bar_chart(frame.set_index("Date")[metric_choice])
        with table_col:
            st.dataframe(frame.set_index("Date"))

        week_consumed = db.get_totals_between(start, end)
        week_goal = goals.effective_weekly()
        st.markdown("**Week to date vs. weekly target**")
        macro_progress(week_consumed, week_goal)

        # ---- Eating pattern: day x meal-slot grid ----
        st.divider()
        st.subheader(f"How you've been eating — last {PATTERN_DAYS} days")
        st.caption("Each cell shows what you logged in that slot, with its calories and protein.")

        pattern_start, pattern_end = db.last_n_days(target_date, PATTERN_DAYS)
        pattern_grid = db.get_meals_by_slot_range(pattern_start, pattern_end)
        slot_stats = db.get_slot_stats(pattern_start, pattern_end)

        # unsafe_allow_html is needed for the <br> that stacks several meals in one
        # cell; every value interpolated in is escaped in pattern_markdown().
        st.markdown(pattern_markdown(pattern_grid, target_date), unsafe_allow_html=True)

        st.markdown("")  # breathing room under the table
        habit_columns = st.columns(len(MEAL_SLOTS))
        for column, slot in zip(habit_columns, MEAL_SLOTS):
            stat = slot_stats[slot]
            days = int(stat["days_logged"])
            with column:
                st.metric(
                    label=SLOT_LABELS[slot],
                    value=f"{days}/{PATTERN_DAYS} days",
                    delta=(f"avg {stat['avg_calories']:,.0f} kcal · {stat['avg_protein_g']:.0f}g P" if days else "never logged"),
                    delta_color="off",
                )

        skipped = [SLOT_LABELS[s] for s in ("breakfast", "lunch", "dinner") if slot_stats[s]["days_logged"] < PATTERN_DAYS / 2]
        if skipped:
            st.caption(
                f"Missing from more than half the days: {', '.join(skipped)}. "
                "Skipped slots are usually where a protein or calorie gap comes from."
            )

        # ---- Recommendations: second LLM call, fed the numbers computed above ----
        st.divider()
        st.subheader("What should I eat next?")
        st.caption(
            f"Python computed the remaining gap from your logged meals and rendered the "
            f"{PATTERN_DAYS}-day pattern above; the model only turns those numbers into "
            "food ideas that fit how you actually eat."
        )

        now = datetime.now()
        st.markdown(
            f"Next up: **{SLOT_LABELS[current_meal_slot(now)].rstrip('s')}**. "
            f"Remaining right now (`{time_of_day_label(now)}`): "
            f"**{remaining.calories:,.0f} kcal · {remaining.protein_g:,.0f}g protein · "
            f"{remaining.carbs_g:,.0f}g carbs · {remaining.fat_g:,.0f}g fat**"
        )

        preferences = st.text_input(
            "Preferences or constraints (optional)",
            placeholder="vegetarian, no dairy, nothing that needs cooking",
        )
        count = st.slider("How many suggestions?", min_value=2, max_value=3, value=3)

        if st.button("Get suggestions", type="primary"):
            try:
                with st.spinner("Thinking about your remaining macros..."):
                    # LLM CALL #2: everything numeric here was calculated in Python.
                    result = recommend_meals(
                        goal=daily_goal,
                        consumed=consumed,
                        remaining=remaining,
                        furthest_behind=behind,
                        meals_today=[m["meal_name"] for m in db.get_meals_for_date(target_date)],
                        weekly_context=weekly_context_line(per_day, daily_goal),
                        eating_pattern=pattern_for_prompt(pattern_grid, target_date),
                        slot_habits=slot_habits_for_prompt(slot_stats, PATTERN_DAYS),
                        preferences=preferences,
                        num_suggestions=count,
                        now=now,
                    )
                st.session_state["suggestions"] = result
            except MissingAPIKeyError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not generate suggestions: {exc}")

        result = st.session_state.get("suggestions")
        if result is not None:
            if result.strategy_note:
                st.info(result.strategy_note)
            for suggestion in result.suggestions:
                with st.container(border=True):
                    st.markdown(f"**{suggestion.name}**")
                    st.write(suggestion.why_it_fits)
                    st.caption(
                        format_macros(
                            MacroTotals(
                                calories=suggestion.calories,
                                protein_g=suggestion.protein_g,
                                carbs_g=suggestion.carbs_g,
                                fat_g=suggestion.fat_g,
                            )
                        )
                    )


# ---------------------------------------------------------------------------
# TAB 3 -- Set goals
# ---------------------------------------------------------------------------

with goals_tab:
    st.subheader("Daily targets")

    # Survives the st.rerun() below so the confirmation is still visible afterwards.
    if st.session_state.pop("goals_saved", False):
        st.success("Goals saved.")

    existing = db.get_goals()
    current_daily = existing.daily if existing else MacroTotals(calories=2200, protein_g=140, carbs_g=220, fat_g=70)

    with st.form("goals_form"):
        c1, c2, c3, c4 = st.columns(4)
        daily_calories = c1.number_input("Calories", min_value=0.0, value=float(current_daily.calories), step=50.0)
        daily_protein = c2.number_input("Protein (g)", min_value=0.0, value=float(current_daily.protein_g), step=5.0)
        daily_carbs = c3.number_input("Carbs (g)", min_value=0.0, value=float(current_daily.carbs_g), step=5.0)
        daily_fat = c4.number_input("Fat (g)", min_value=0.0, value=float(current_daily.fat_g), step=5.0)

        st.markdown("---")
        use_weekly = st.checkbox(
            "Set explicit weekly totals",
            value=existing is not None and existing.weekly is not None,
            help="Leave off and the weekly target is simply 7 x your daily target.",
        )
        current_weekly = (existing.weekly if existing and existing.weekly else current_daily.scaled(7))

        w1, w2, w3, w4 = st.columns(4)
        weekly_calories = w1.number_input(
            "Weekly calories", min_value=0.0, value=float(current_weekly.calories), step=100.0, disabled=not use_weekly
        )
        weekly_protein = w2.number_input(
            "Weekly protein (g)", min_value=0.0, value=float(current_weekly.protein_g), step=10.0, disabled=not use_weekly
        )
        weekly_carbs = w3.number_input(
            "Weekly carbs (g)", min_value=0.0, value=float(current_weekly.carbs_g), step=10.0, disabled=not use_weekly
        )
        weekly_fat = w4.number_input(
            "Weekly fat (g)", min_value=0.0, value=float(current_weekly.fat_g), step=10.0, disabled=not use_weekly
        )

        if st.form_submit_button("Save goals", type="primary"):
            db.save_goals(
                Goals(
                    daily=MacroTotals(
                        calories=daily_calories,
                        protein_g=daily_protein,
                        carbs_g=daily_carbs,
                        fat_g=daily_fat,
                    ),
                    weekly=MacroTotals(
                        calories=weekly_calories,
                        protein_g=weekly_protein,
                        carbs_g=weekly_carbs,
                        fat_g=weekly_fat,
                    )
                    if use_weekly
                    else None,
                )
            )
            # The Dashboard tab is rendered earlier in this script run, so it already
            # read the *old* goals. Rerun so every tab reflects the new targets now.
            st.session_state["goals_saved"] = True
            st.rerun()

    if existing is not None:
        st.caption(f"Current daily goal: {format_macros(existing.daily)}")
        st.caption(
            f"Effective weekly goal: {format_macros(existing.effective_weekly())}"
            + ("" if existing.weekly else "  (derived: 7 × daily)")
        )
