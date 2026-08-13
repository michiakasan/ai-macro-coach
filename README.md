# 🥗 MacroTrack

A macro-goal food tracker. Set daily (and optionally weekly) targets for calories,
protein, carbs and fat; log meals in plain language; and get meal suggestions chosen
specifically to close the gap between what you've eaten and what you still need.

Built with Python, LangChain (LCEL + structured output), OpenAI `gpt-4o-mini`,
Streamlit and SQLite.

---

## Setup

```bash
# 1. clone / cd into the project
cd food-recommendation

# 2. create a virtualenv
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. add your OpenAI API key
cp .env.example .env
$EDITOR .env                      # set OPENAI_API_KEY=sk-...

# 5. run it
streamlit run app.py
```

The app opens at http://localhost:8501. On first run it creates `macrotrack.db`
in the project directory — that's where your food log lives, so it survives
restarts and persists across the week.

**First steps in the UI:** open **Set Goals** and save your daily targets, then log a
meal on the **Log Meal** tab, then check the **Dashboard**.

---

## How it works

### The one architectural idea

The LLM is used for two things it's genuinely good at, and deliberately kept away
from a third thing it's bad at.

```
  ┌─ LLM job #1 ────────────────────────────────────────────────┐
  │  "two eggs and toast with butter"                           │
  │            ↓  chains.estimate_meal()                        │
  │  MealEstimate(calories=340, protein_g=19, carbs_g=27, ...)  │
  └─────────────────────────────────────────────────────────────┘
                       ↓  user confirms / corrects
                       ↓  database.add_meal()  → SQLite row
  ┌─ PYTHON ONLY ───────────────────────────────────────────────┐
  │  SELECT SUM(protein_g) ... WHERE log_date = ?               │
  │            ↓  database.get_totals_for_date()                │
  │  consumed = MacroTotals(...)                                │
  │            ↓  consumed.remaining_from(goal)                 │
  │  remaining = MacroTotals(protein_g=55, calories=420, ...)   │
  │                                                             │
  │  + the 6-day day × slot grid and per-slot averages          │
  │    (database.get_meals_by_slot_range / get_slot_stats)      │
  └─────────────────────────────────────────────────────────────┘
                       ↓  the finished numbers are passed in as context
  ┌─ LLM job #2 ────────────────────────────────────────────────┐
  │  chains.recommend_meals(remaining=..., eating_pattern=...)  │
  │            ↓                                                │
  │  SuggestionSet(suggestions=[...])                           │
  └─────────────────────────────────────────────────────────────┘
```

**Why the middle box is not an LLM call:** running totals are cumulative state. A
model that mis-adds a single meal poisons every subsequent number, and the error is
invisible — the output still *looks* like a plausible total. `SUM()` over stored rows
is exact, cheap, instant, and testable. So the model estimates individual meals and
writes suggestion copy; Python owns every running total and every gap.

The recommendation prompt reinforces this: it tells the model outright that the
figures it's been handed were calculated by the application and must not be
recalculated or second-guessed.

### File map

| File | Responsibility |
|---|---|
| `app.py` | Streamlit UI — three tabs (Log Meal / Dashboard / Set Goals). Wiring only, no nutrition logic. |
| `chains.py` | Both LangChain chains: prompt templates, model config, `.with_structured_output(...)`, and the helpers that pre-render Python-computed numbers for the prompt. |
| `database.py` | SQLite schema and every query. The **only** place consumed totals are produced (via `SUM`). |
| `models.py` | Pydantic models. LLM-facing output schemas *and* the pure-Python macro arithmetic (`remaining_from`, `percent_of`, `furthest_behind`). |

### LangChain specifics

Both chains are LCEL pipelines:

```python
prompt = ChatPromptTemplate.from_messages([("system", ...), ("human", ...)])
chain  = prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(MealEstimate)
estimate = chain.invoke({"meal_description": ..., "today": ...})   # -> MealEstimate instance
```

`.with_structured_output(PydanticModel)` is the current replacement for the old
`PydanticOutputParser` + `{format_instructions}` pattern. LangChain converts the
Pydantic class into a JSON schema, sends it to the OpenAI API as the required
response format, and validates the reply back into a typed object — so a malformed
response raises instead of quietly returning prose.

Temperatures differ on purpose: `0` for estimation (the same meal should score the
same every time) and `0.6` for recommendations (otherwise it's chicken and broccoli
every single evening).

### Prompt engineering notes

**Meal estimation** (`MEAL_ESTIMATION_SYSTEM` in `chains.py`)
- Never refuse, never ask a follow-up — always make a best-effort estimate from
  typical portion sizes when the user is vague.
- Concrete reference portions are given in the prompt (a chicken breast is ~150–200g,
  a slice of bread ~30g) so estimates don't drift.
- Exact user-specified quantities must be used verbatim rather than overridden.
- The `assumptions` field is mandatory and must be specific — *"assumed ~150g chicken,
  1 cup cooked rice, 1 tsp oil"*, not "assumed a normal portion". It's shown in the UI
  so the user can spot a bad guess before saving.
- A 4/4/9 kcal-per-gram consistency rule keeps the four numbers coherent.

**Recommendations** (`RECOMMENDATION_SYSTEM` in `chains.py`)
- Opens by declaring the supplied numbers are app-calculated ground truth and must not
  be recomputed.
- Reasons about the *specific* gap: anchor on the macro furthest behind, respect the
  remaining calorie ceiling (lots of protein left + few calories ⇒ lean sources), and
  avoid any macro already over target.
- Time of day steers the format — a real meal in the evening, a light snack near
  bedtime, breakfast food in the morning. Python decides the upcoming slot
  (`current_meal_slot`) and names it in the prompt rather than leaving the model to
  reason about the clock.
- The 6-day eating pattern is passed in as context, with instructions to fit the user's
  actual habits and tastes, avoid repeating something eaten in the last day or two, and
  name the habit behind the gap (a skipped breakfast, carb-heavy dinners) in
  `strategy_note`.
- `why_it_fits` must cite the actual remaining numbers, which makes generic "eat
  something healthy" answers hard to produce.

### Data model

```
goals(id=1, daily_*, weekly_* nullable, updated_at)    -- one row; weekly NULL ⇒ 7 × daily
meals(id, log_date, logged_at, meal_name, meal_slot,
      calories, protein_g, carbs_g, fat_g,
      estimated, assumptions, raw_input)               -- one row per logged meal
```

`meal_slot` is `breakfast` / `lunch` / `dinner` / `snack` — the columns of the pattern
table. The model infers it from the description ("...for lunch") or the current time,
and the user can correct it in the same confirm step as the macros.

`estimated` records provenance: `1` when the numbers came straight from the model,
`0` when the user hand-corrected them in the confirm step. `raw_input` keeps the
original free-text description, so an estimate can always be traced back to what was
actually typed.

`init_db()` runs additive migrations (`database._migrate`) after creating the schema,
because `CREATE TABLE IF NOT EXISTS` won't add a column to a database that already
exists — a log created before `meal_slot` existed picks it up on next launch, with old
rows defaulting to `snack`.

---

## Features

- **Goal setup** — daily calorie/protein/carb/fat targets, editable any time, with
  optional explicit weekly totals (otherwise weekly = 7 × daily).
- **Natural-language logging** — describe a meal however you like; the model estimates
  the macros and states its portion assumptions.
- **Confirm-before-save** — every estimate is shown in editable fields first. Nothing
  reaches the database until you accept it, and edits are flagged as user-adjusted.
- **Dashboard** — today's totals vs. goal as metrics and progress bars, an explicit
  callout for the macro furthest behind, a 7-day chart and table, and week-to-date
  totals against the weekly target.
- **Eating-pattern table** — the last 6 days as a grid of day × meal slot
  (Breakfast / Lunch / Dinner / Snacks), each cell showing what you logged with its
  calories and protein, plus a daily total. Underneath: how often you actually eat each
  slot ("Breakfast: 2/6 days") with its average macros, and a callout for slots you skip
  more often than not — usually where the gap is coming from.
- **Gap-aware suggestions** — 2–3 suggestions generated from the remaining macros, the
  upcoming meal slot, and that 6-day pattern, so they fit how you actually eat rather
  than repeating yesterday's dinner. Each comes with a rough macro estimate and an
  explanation tied to your actual numbers. Accepts free-text constraints ("vegetarian,
  nothing that needs cooking").

## Notes and limitations

- Macro figures are LLM estimates, not lab measurements. Treat them as directional,
  and correct them in the confirm step when you know better.
- Single-user by design: the goals table is pinned to one row and there's no auth.
- Each estimate and each recommendation is one `gpt-4o-mini` call; costs are
  negligible at personal-tracking volume.
