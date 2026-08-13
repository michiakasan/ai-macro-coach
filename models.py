"""
Pydantic models for MacroTrack.

Two categories live here:

1. LLM-facing schemas (`MealEstimate`, `MealSuggestion`, `SuggestionSet`).
   These are handed to LangChain's `.with_structured_output(...)`, which turns the
   model class into a JSON schema for the OpenAI API so the response comes back as
   a validated object instead of free text we'd have to parse.

2. Plain data/arithmetic models (`Goals`, `MacroTotals`).
   These never touch the LLM. All totals and gaps are computed here in Python from
   database rows -- the LLM is never asked to add numbers up. That separation is
   deliberate: LLMs are good at "how much protein is in 200g of chicken?" and bad at
   reliably summing 14 meals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# The four macros we track, in a fixed order used for iteration in the UI.
MACRO_FIELDS = ("calories", "protein_g", "carbs_g", "fat_g")

# Human-readable labels + units for the dashboard.
MACRO_LABELS = {
    "calories": ("Calories", "kcal"),
    "protein_g": ("Protein", "g"),
    "carbs_g": ("Carbs", "g"),
    "fat_g": ("Fat", "g"),
}

# Which slot of the day a meal belongs to. Used as the columns of the eating-pattern
# table, and as context for recommendations ("it's 18:00, so plan dinner").
MealSlot = Literal["breakfast", "lunch", "dinner", "snack"]
MEAL_SLOTS: tuple[MealSlot, ...] = ("breakfast", "lunch", "dinner", "snack")
SLOT_LABELS = {
    "breakfast": "Breakfast",
    "lunch": "Lunch",
    "dinner": "Dinner",
    "snack": "Snacks",
}


# ---------------------------------------------------------------------------
# 1. LLM-facing schemas
# ---------------------------------------------------------------------------


class MealEstimate(BaseModel):
    """Structured macro estimate for one free-text meal description.

    This is the output schema of the meal-estimation chain (see chains.py).
    Field descriptions are not just documentation -- they are sent to the model
    as part of the JSON schema, so they double as per-field prompt instructions.
    """

    meal_name: str = Field(
        description=(
            "Short, clean name for the meal, e.g. 'Grilled chicken breast with rice'. "
            "Do not include quantities here."
        )
    )
    meal_slot: MealSlot = Field(
        description=(
            "Which slot of the day this belongs to. Use what the user actually says "
            "('...for lunch' -> lunch). If they don't say, infer from the current time "
            "you are given and from the food itself (cereal/eggs/toast -> breakfast). "
            "Use 'snack' for small between-meal items or when it genuinely isn't a "
            "main meal. The user can correct this before it is saved."
        )
    )
    calories: float = Field(ge=0, description="Total estimated calories (kcal) for the whole meal.")
    protein_g: float = Field(ge=0, description="Total estimated protein in grams.")
    carbs_g: float = Field(ge=0, description="Total estimated carbohydrates in grams.")
    fat_g: float = Field(ge=0, description="Total estimated fat in grams.")
    assumptions: str = Field(
        default="",
        description=(
            "One or two sentences naming the portion sizes and preparation methods you "
            "assumed where the user did not specify them, e.g. "
            "'Assumed a medium portion of ~150g chicken and 1 cup cooked rice; "
            "assumed grilled with ~1 tsp oil.' Empty string only if the user specified "
            "every quantity exactly."
        ),
    )
    estimated: bool = Field(
        default=True,
        description=(
            "Always true for LLM output: these are model estimates, not lab-verified "
            "values. Flipped to false by the app only if the user hand-edits the numbers."
        ),
    )

    def totals(self) -> "MacroTotals":
        """Convert this single meal into a MacroTotals so it can be summed/compared."""
        return MacroTotals(
            calories=self.calories,
            protein_g=self.protein_g,
            carbs_g=self.carbs_g,
            fat_g=self.fat_g,
        )


class MealSuggestion(BaseModel):
    """One recommended meal or snack, output by the recommendation chain."""

    name: str = Field(description="Name of the suggested meal or snack, with portion size.")
    why_it_fits: str = Field(
        description=(
            "Two sentences max explaining how this closes the *specific* remaining gap "
            "you were given. Reference the actual remaining numbers, e.g. "
            "'Covers ~40g of your remaining 55g protein while only using 12g of carbs.'"
        )
    )
    calories: float = Field(ge=0, description="Rough calories for the suggested portion.")
    protein_g: float = Field(ge=0, description="Rough protein in grams for the suggested portion.")
    carbs_g: float = Field(ge=0, description="Rough carbs in grams for the suggested portion.")
    fat_g: float = Field(ge=0, description="Rough fat in grams for the suggested portion.")


class SuggestionSet(BaseModel):
    """Wrapper so the model returns a list of suggestions in one structured call."""

    suggestions: list[MealSuggestion] = Field(
        min_length=1,
        max_length=4,
        description="Two or three suggestions, ordered best-fit first.",
    )
    strategy_note: str = Field(
        default="",
        description=(
            "One sentence on the overall strategy for the rest of the day, given the "
            "time and the remaining macros. E.g. 'You're 60g of protein behind with only "
            "400 kcal left, so lean protein with minimal fat is the priority.'"
        ),
    )


# ---------------------------------------------------------------------------
# 2. Plain data models -- all arithmetic happens here, never in the LLM
# ---------------------------------------------------------------------------


class MacroTotals(BaseModel):
    """A bag of four macro numbers: a goal, a consumed total, or a remaining gap."""

    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0

    def __add__(self, other: "MacroTotals") -> "MacroTotals":
        return MacroTotals(**{f: getattr(self, f) + getattr(other, f) for f in MACRO_FIELDS})

    def remaining_from(self, goal: "MacroTotals") -> "MacroTotals":
        """Gap = goal - consumed (self). Negative values mean the goal was exceeded.

        We keep negatives rather than clamping at zero so the recommendation prompt can
        see "you are 300 kcal *over*" and react to it.
        """
        return MacroTotals(**{f: getattr(goal, f) - getattr(self, f) for f in MACRO_FIELDS})

    def scaled(self, factor: float) -> "MacroTotals":
        """Scale every macro (used to derive weekly targets from daily ones)."""
        return MacroTotals(**{f: getattr(self, f) * factor for f in MACRO_FIELDS})

    def percent_of(self, goal: "MacroTotals") -> dict[str, float]:
        """Fraction of goal hit per macro, e.g. {'protein_g': 0.61, ...}.

        A goal of 0 is treated as already complete (1.0) so it never shows as "behind".
        """
        out: dict[str, float] = {}
        for field in MACRO_FIELDS:
            target = getattr(goal, field)
            out[field] = getattr(self, field) / target if target > 0 else 1.0
        return out

    def furthest_behind(self, goal: "MacroTotals") -> str | None:
        """Name of the macro with the lowest completion ratio, or None if all met.

        Pure Python: the dashboard's "what am I most behind on?" callout and the
        recommendation prompt both use this, so the answer is consistent and testable.
        """
        pct = self.percent_of(goal)
        behind = {f: p for f, p in pct.items() if p < 1.0}
        if not behind:
            return None
        return min(behind, key=behind.get)


class Goals(BaseModel):
    """The user's targets. Daily is required; weekly is optional and overrides 7x daily."""

    daily: MacroTotals
    weekly: MacroTotals | None = None

    def effective_weekly(self) -> MacroTotals:
        """Weekly target: explicit if the user set one, otherwise 7 x the daily target."""
        return self.weekly if self.weekly is not None else self.daily.scaled(7)
