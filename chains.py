"""
LangChain orchestration for MacroTrack.

Two chains live here, and the split between them is the heart of the architecture:

  CHAIN 1 -- estimate_meal()
      free text ("two eggs and toast with butter") -> MealEstimate
      The LLM does what it's actually good at: mapping a fuzzy food description to
      plausible macro numbers for a *single* item, and stating the portion
      assumptions it made.

  ... then Python takes over. app.py stores the meal, database.py SUMs the rows,
      and models.MacroTotals.remaining_from() computes goal - consumed. No LLM is
      involved in any of that arithmetic, because a model that mis-adds one number
      corrupts the whole week's tracking.

  CHAIN 2 -- recommend_meals()
      the *already-computed* gap -> SuggestionSet
      The LLM is handed the remaining macros as finished numbers and asked only to
      generate food ideas that fit them. It never sees the raw meal log to add up.

Both chains are built with LangChain Expression Language (`prompt | llm`) and use
`.with_structured_output(PydanticModel)`, the current replacement for the old
output-parser-plus-format-instructions pattern. Under the hood LangChain converts the
Pydantic class into a JSON schema, passes it to the OpenAI API as a tool/response
format, and validates the reply back into the model instance -- so a malformed
response raises instead of silently returning garbage text.
"""

from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import AuthenticationError

from models import MACRO_LABELS, MacroTotals, MealEstimate, SuggestionSet

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class MissingAPIKeyError(RuntimeError):
    """Raised when OPENAI_API_KEY is absent, placeholder, or rejected by OpenAI.

    The UI prints this straight to the user, so the message has to say what to fix.
    """


# The value shipped in .env.example. Copying that file without editing it is the most
# likely first-run mistake, and it fails as an opaque 401 unless we catch it here.
PLACEHOLDER_KEYS = {"sk-your-key-here", "sk-...", ""}


def _make_llm(temperature: float) -> ChatOpenAI:
    """Build the chat model. Key comes from OPENAI_API_KEY (loaded by python-dotenv)."""
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    if key in PLACEHOLDER_KEYS or "your-key" in key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is still the placeholder from .env.example. "
            "Edit .env and replace it with a real key from "
            "https://platform.openai.com/api-keys"
        )
    return ChatOpenAI(model=DEFAULT_MODEL, temperature=temperature)


def _invoke(chain: Runnable, payload: dict) -> object:
    """Run a chain, translating OpenAI auth failures into a message worth reading.

    Everything else (rate limits, network, schema validation) propagates unchanged --
    app.py shows those verbatim, since the detail is what makes them debuggable.
    """
    try:
        return chain.invoke(payload)
    except AuthenticationError as exc:
        raise MissingAPIKeyError(
            "OpenAI rejected the API key in your .env (401). Check that it is current "
            "and has not been revoked: https://platform.openai.com/api-keys"
        ) from exc


# ---------------------------------------------------------------------------
# CHAIN 1: free-text meal -> structured macro estimate
# ---------------------------------------------------------------------------

MEAL_ESTIMATION_SYSTEM = """You are a nutrition estimation assistant for a macro-tracking app.

Given a free-text description of something the user ate, estimate the macronutrients for \
the ENTIRE described meal (all items combined into one set of numbers).

Rules:
1. The user will often be vague about quantities. Never refuse and never ask a follow-up \
question -- make a best-effort estimate using typical real-world portion sizes \
(e.g. one chicken breast is ~150-200g cooked, a "bowl of rice" is ~1 cup cooked / ~200g, \
a slice of bread is ~30g, a pat of butter is ~7g).
2. Account for cooking method when it materially changes the numbers: fried adds oil, \
grilled adds little, creamy sauces add fat.
3. If the user DOES specify a quantity, use it exactly and do not substitute your own.
4. Always fill in the `assumptions` field with the specific portion sizes and preparation \
you assumed, so the user can spot and correct a bad guess. Be concrete: \
"assumed ~150g chicken breast, 1 cup cooked white rice, 1 tsp oil" -- not "assumed a \
normal portion".
5. Use standard nutrition data as your reference point and keep the four numbers \
internally consistent: roughly 4 kcal per gram of protein, 4 per gram of carbs, \
9 per gram of fat. The calorie figure should be close to that sum.
6. These are estimates, not measurements. Do not over-claim precision -- round to \
sensible whole numbers.
7. Set `meal_slot` from what the user says ("...for lunch" -> lunch). If they don't say, \
infer it from the current time below and from the food itself. Small between-meal items \
are 'snack'.

It is currently {now}. Only estimate; the app handles all the tracking totals."""

MEAL_ESTIMATION_HUMAN = """The user ate: {meal_description}"""


@lru_cache(maxsize=1)
def get_meal_estimation_chain() -> Runnable:
    """Build (and cache) the LCEL chain: prompt | model-with-structured-output.

    temperature=0 -- estimating macros is a lookup-ish task where we want the same
    answer for the same input, not creative variety.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", MEAL_ESTIMATION_SYSTEM),
            ("human", MEAL_ESTIMATION_HUMAN),
        ]
    )
    # .with_structured_output() replaces the deprecated PydanticOutputParser +
    # {format_instructions} approach: schema enforcement happens at the API level.
    llm = _make_llm(temperature=0).with_structured_output(MealEstimate)
    return prompt | llm


def estimate_meal(meal_description: str) -> MealEstimate:
    """Run chain 1. Returns a validated MealEstimate the user can then edit."""
    chain = get_meal_estimation_chain()
    return _invoke(
        chain,
        {
            "meal_description": meal_description.strip(),
            "now": datetime.now().strftime("%A, %d %B %Y, %H:%M"),
        },
    )


# ---------------------------------------------------------------------------
# CHAIN 2: pre-computed gap -> meal suggestions
# ---------------------------------------------------------------------------

RECOMMENDATION_SYSTEM = """You are a macro-aware meal planner. Your only job is to suggest \
food that closes a specific, already-calculated nutritional gap.

CRITICAL: every number you are given below was calculated by the application from the \
user's logged meals. Do not recalculate, re-derive, or second-guess them, and do not sum \
anything yourself. Treat the remaining-macro figures as ground truth and plan against them.

How to choose suggestions:
- Anchor on the macro that is furthest behind. If protein is the gap, lead with \
high-protein, low-calorie-density options.
- Respect the calorie ceiling. If there is a lot of protein left but few calories, \
suggest lean sources (white fish, egg whites, non-fat Greek yoghurt, chicken breast, \
whey) rather than fatty ones.
- If a macro is already over target, actively avoid adding more of it and say so.
- Weight the suggestions by time of day: a full meal in the late afternoon or evening, \
lighter snacks close to bedtime, breakfast-appropriate food in the morning. Aim them at \
the upcoming meal slot named below.
- Sizes must be realistic and specific ("200g non-fat Greek yoghurt with 20g almonds"), \
not vague ("some yoghurt").
- The suggestions should together be capable of closing most of the gap, but no single \
suggestion should blow past the remaining calories.

USE THE EATING PATTERN TABLE. You are shown what the user actually ate in each slot over \
recent days, plus per-slot averages, all computed by the app. Use it to:
- Fit their habits and tastes: suggest things that resemble food they already eat, rather \
than a diet plan they will ignore.
- Avoid repeating something they have eaten in the last day or two, unless it is an \
unusually good fit -- variety matters.
- Name the habit that is causing the gap when there is one, e.g. a repeatedly skipped or \
tiny breakfast, or carb-heavy dinners. Say it in `strategy_note`, concretely and without \
lecturing.

In `why_it_fits`, cite the actual remaining numbers you were given so the user can see \
the connection. Give exactly {num_suggestions} suggestions, best fit first."""

RECOMMENDATION_HUMAN = """Here is the user's current state, calculated by the app:

CURRENT TIME: {current_time} ({time_of_day})
UPCOMING MEAL SLOT: {upcoming_slot}

DAILY GOAL:      {goal_line}
CONSUMED SO FAR: {consumed_line}
REMAINING TODAY: {remaining_line}

FURTHEST BEHIND: {furthest_behind}
MEALS ALREADY LOGGED TODAY: {meals_today}

WEEKLY CONTEXT (last 7 days): {weekly_context}

EATING PATTERN -- what they actually ate by slot, most recent day first:
{eating_pattern}

PER-SLOT HABITS OVER THAT WINDOW:
{slot_habits}

USER PREFERENCES / CONSTRAINTS: {preferences}

Suggest what they should eat next to close the remaining gap above."""


@lru_cache(maxsize=1)
def get_recommendation_chain() -> Runnable:
    """Build (and cache) the LCEL chain for suggestions.

    temperature=0.6 -- here we *want* variation, otherwise the user gets the same
    chicken-and-broccoli answer every single evening.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RECOMMENDATION_SYSTEM),
            ("human", RECOMMENDATION_HUMAN),
        ]
    )
    llm = _make_llm(temperature=0.6).with_structured_output(SuggestionSet)
    return prompt | llm


# --- helpers that pre-render Python-computed numbers for the prompt ---------


def format_macros(totals: MacroTotals) -> str:
    """'1,240 kcal | 78g protein | 130g carbs | 42g fat' -- a compact prompt-friendly line."""
    return (
        f"{totals.calories:,.0f} kcal | "
        f"{totals.protein_g:.0f}g protein | "
        f"{totals.carbs_g:.0f}g carbs | "
        f"{totals.fat_g:.0f}g fat"
    )


def format_remaining(remaining: MacroTotals) -> str:
    """Like format_macros, but spells out overshoot so the model can't misread a minus sign."""
    parts = []
    for field, (label, unit) in MACRO_LABELS.items():
        value = getattr(remaining, field)
        amount = f"{abs(value):,.0f} kcal" if field == "calories" else f"{abs(value):,.0f}{unit} {label.lower()}"
        parts.append(f"{amount} left" if value >= 0 else f"{amount} OVER target")
    return " | ".join(parts)


def time_of_day_label(now: datetime) -> str:
    """Coarse bucket used to steer the suggestions (snack vs. full meal)."""
    hour = now.hour
    if hour < 10:
        return "morning"
    if hour < 14:
        return "midday"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "late night"


def current_meal_slot(now: datetime) -> str:
    """Which slot the user is most likely about to eat, from the wall clock.

    Decided in Python so the recommendation prompt is told the slot outright rather
    than left to reason about the time itself.
    """
    hour = now.hour
    if hour < 10:
        return "breakfast"
    if hour < 15:
        return "lunch"
    if hour < 17:
        return "snack"
    if hour < 21:
        return "dinner"
    return "snack"


def recommend_meals(
    *,
    goal: MacroTotals,
    consumed: MacroTotals,
    remaining: MacroTotals,
    furthest_behind: str | None,
    meals_today: list[str],
    weekly_context: str = "not available",
    eating_pattern: str = "not available",
    slot_habits: str = "not available",
    preferences: str = "",
    num_suggestions: int = 3,
    now: datetime | None = None,
) -> SuggestionSet:
    """Run chain 2.

    Note the signature: `remaining` and `furthest_behind` are *inputs*, not something
    this function or the model works out. app.py computes them from database totals
    (models.MacroTotals.remaining_from) and passes them in. The LLM only ever reads
    finished numbers.
    """
    now = now or datetime.now()
    chain = get_recommendation_chain()
    label = MACRO_LABELS[furthest_behind][0] if furthest_behind else "nothing -- all targets met"

    return _invoke(
        chain,
        {
            "current_time": now.strftime("%A %H:%M"),
            "time_of_day": time_of_day_label(now),
            "upcoming_slot": current_meal_slot(now),
            "goal_line": format_macros(goal),
            "consumed_line": format_macros(consumed),
            "remaining_line": format_remaining(remaining),
            "furthest_behind": label,
            "meals_today": ", ".join(meals_today) if meals_today else "nothing logged yet today",
            "weekly_context": weekly_context,
            "eating_pattern": eating_pattern,
            "slot_habits": slot_habits,
            "preferences": preferences.strip() or "none given",
            "num_suggestions": num_suggestions,
        }
    )
