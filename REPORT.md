# REPORT — Steam Deal Assistant

CPSC 254 Final Project · William Wang · gpt-4o-mini · 12 labeled test cases

---

## 1. What & why

The Steam Deal Assistant helps PC gamers on a fixed budget decide whether to
**Buy**, **Wait**, or **Skip** a discounted game. The user types a
natural-language message — for example *"I have $20, I love co-op shooters,
already own Apex"* — and the app responds with one to three picks from a local
catalog of 20 Steam games, each tagged with a verdict and reasoning that names
the current price, the genre, and how the pick fits the user's stated
preferences. The intended user is myself and friends who already own several
hundred dollars of unplayed Steam games and want a sanity check before clicking
Add to Cart.

The hard part of the AI behavior is **grounding**. Three failure modes show up
constantly:

1. **Hallucinated games.** Recent Steam releases live in the model's training
   data and it wants to recommend them even when they aren't in the catalog.
2. **Ungrounded prices.** A single-call prompt has the model freely write
   prices in its reasoning text that don't match the catalog.
3. **Verdict drift.** Without explicit budget rules, "Buy" appears against
   over-budget games with positive reasoning attached.

A single extraction call cannot fix any of these because there is no source of
truth bound to the call. The project therefore splits the AI into a three-stage
pipeline (extraction → tool-calling search → recommendation) plus a
deterministic verification layer, and it iterates on the retry policy.

---

## 2. Iterations

Four versions were measured against the same 12 test cases in
`eval/test_cases.json`. Eval data lives in `eval/results_v{1,2,3,4}_*.json`.

### V1 — Single recommendation prompt (baseline)

- **Change.** One LLM call. The user message is appended to a system prompt
  that includes the entire catalog as JSON. No preference extraction stage,
  no tool call, no verification. (`app/pipeline_v1.py`)
- **Motivating example.** `tc09_overbudget_should_not_buy`: with a $30 budget
  and a request for a souls-like, the model recommended Elden Ring at
  $41.99 *and* invented a non-catalog game in the same response, which is the
  exact pattern Project 4's "text-in → JSON-out" call produces.
- **Delta.** recommendation_pass_rate = **8/12 = 66.7%**. valid_game = 83.3%
  (one hallucinated game), expectation_match = 83.3%.
- **Conclusion.** The metric is misleadingly OK because most test cases are
  easy; the hallucination on `tc09` and the missing keyword "12.49" in
  `tc02_hades_target_price_wait` are concrete signs the model is freelancing
  on prices and titles. Next: ground the candidate set in code rather than
  prose.

### V2 — Three-stage pipeline with function calling

- **Change.** Split the call into three stages: a preference-extraction call
  produces a structured `UserPreferences` JSON object; the model then calls a
  `search_games` tool that runs deterministically against `games.json`; a
  final call writes Buy/Wait/Skip rationales using only the returned
  candidates. (`app/pipeline_v2.py`, with shared stages in `app/pipeline.py`)
- **Motivating example.** V1's `tc09` hallucination above. After V2,
  hallucinations drop to zero — every recommendation comes from the tool's
  output. valid_game climbs from 83.3% to **100%**.
- **Delta.** recommendation_pass_rate = **8/12 = 66.7%** (unchanged), but
  the sub-metric mix moved: valid_game 83.3% → 100%, expectation_match
  83.3% → 66.7%.
- **Conclusion.** Structure killed hallucinations but introduced a new
  failure: the model's chosen tool filters were sometimes too strict, hiding
  the *right* game from the recommendation step. `tc02` failed because
  `search_games(genres=["roguelike"], max_price=15)` returned Dead Cells but
  not Hades (Hades is $24.99). `tc07` and `tc09` failed for the same reason.
  Next: catch verifier-detectable failures and retry.

### V3 — Verification with a naive retry

- **Change.** Added `verify_recommendations` (`app/pipeline.py:326`). It
  rejects outputs that recommend non-catalog games, mark over-budget games as
  Buy, or omit price/genre/user-fit cues from the reason. On any violation,
  stage C is re-run once with the violation text appended, and the retry's
  output **unconditionally replaces** the original. (`app/pipeline_v3.py`)
- **Motivating example.** `tc04_singleplayer_only`: the first call produced
  good single-player picks but worded the reasons in a way the genre-keyword
  check missed. The retry was prompted to fix the reasons but instead
  returned an empty recommendation list, and the unconditional replacement
  wiped out a working answer. Verification then reported "no recommendations
  returned."
- **Delta.** recommendation_pass_rate **dropped from 8/12 to 7/12 (66.7% →
  58.3%)**. valid_game 100% → 91.7%; explanation_complete 100% → 91.7%.
- **Conclusion.** Adding a retry without a fallback strategy traded a small
  number of borderline-correct outputs for total failures. The metric moved
  *down* exactly because the retry over-corrected. Next: only adopt a retry
  when it actually improves on the original.

### V4 — Smart retry-with-fallback

- **Change.** The retry is still triggered by any verification violation, but
  its result is only adopted when it (a) contains at least one recommendation
  and (b) has fewer or equal violations than the original. Otherwise the
  original response is returned with `verification.retried = true` so the UI
  can flag it. (`app/pipeline.py:395`)
- **Motivating example.** Replaying `tc04_singleplayer_only`: the empty
  retry that broke V3 is now rejected because `len(retry_recs) == 0`, and the
  original four single-player picks are returned instead. Case passes.
- **Delta.** recommendation_pass_rate **recovered from 7/12 to 8/12 (58.3% →
  66.7%)**. All four per-call sub-metrics are back to 100%; only
  expectation_match remains at 66.7%.
- **Conclusion.** The fallback rule is small but it cleanly separates two
  concerns the verifier was conflating: *the model's output is invalid*
  versus *the retry produced something better*. The remaining four failures
  (`tc01`, `tc02`, `tc07`, `tc09`) all share one shape: the right game
  exists in the catalog but `search_games` was given filters by the model
  that excluded it. Next, I would either widen the candidate set
  automatically when a user mentions a specific game by name (force-include
  it), or run two parallel `search_games` calls — one strict, one relaxed —
  and merge the candidates before stage C.

---

## 3. Code walkthrough

Tracing one user action — *"I have $20 and I love co-op shooters, I already
own Apex"* — through the codebase:

1. The browser submits the form in `frontend/app.js:130` (the `submit`
   handler), which posts JSON to `/api/recommend`.
2. FastAPI routes the request to `recommend_endpoint` in `app/main.py:21`,
   validating the body against `RecommendRequest` (`app/schemas.py:19`) and
   rejecting empty messages before any LLM call (`app/main.py:23`).
3. `pipeline.run` in `app/pipeline.py:384` orchestrates three OpenAI calls:
   - `extract_preferences` (`app/pipeline.py:73`) returns
     `UserPreferences(budget=20.0, genres=["co-op","fps"], …)` using a
     JSON-mode response with temperature 0 for stability.
   - `select_candidates` (`app/pipeline.py:156`) attaches the `search_games`
     tool definition (`app/pipeline.py:115`) and lets the model pick filters.
     Each tool call runs through `game_data.search_games`
     (`app/game_data.py:32`), which sorts by review percent then discount.
     The loop is capped at three rounds to prevent cost runaway.
   - `recommend` (`app/pipeline.py:282`) writes structured Buy/Wait/Skip
     verdicts using only the candidates returned by stage B.
4. `verify_recommendations` (`app/pipeline.py:326`) checks catalog
   membership, budget compliance on Buy verdicts, and that each reason
   references a price marker, a catalog genre, and a user-fit token. On
   violation, the smart-retry block (`app/pipeline.py:395`) runs stage C
   once more and adopts the retry only when it has at least one rec and
   reduces the violation count.
5. The response flows back to `app.js:120`, which renders cards with verdict
   badges, prices, and a small note when the verifier had to retry
   (`app.js:74`).

**Design decision — function calling instead of a single grounded prompt.**
I split candidate selection from recommendation and used the OpenAI tool API
so the model never sees the entire catalog at once. This forces it to commit
to a deterministic catalog slice rather than free-associating from training
data.

**Rejected alternative — paste the full catalog into one prompt with strict
"only choose from below" instructions.** Simpler and fewer round trips, but
V1 above shows it still hallucinates titles even with explicit constraints,
and the design does not scale: at 200 games the prompt becomes most of the
context window, and at 2,000 games it stops fitting. The three-stage shape
also makes verification meaningful — the verifier can compare against the
exact slice the model was given, not the whole catalog.

---

## 4. AI disclosure & safety

I built this project with **Kiro** as my coding assistant. Three concrete
moments where it failed and how I recovered:

1. **Empty-retry blind replacement (V3 bug).** Kiro wrote the V3 retry to
   unconditionally overwrite the original response. The eval caught it on
   `tc04` (no recommendations returned, retried = true). I diagnosed it by
   reading the V3 results JSON, then asked Kiro to rewrite the retry block
   with an `len(retry_recs) > 0` guard and a "fewer violations" check.
   That became V4.
2. **Misplaced `.env` file.** Kiro's first-pass instructions had me put the
   environment file in the project root, but I created it inside `frontend/`
   by accident. Kiro caught it on the next message, moved the file with
   `Move-Item`, and tightened `.gitignore` to `**/.env` so the same mistake
   wouldn't slip past Git.
3. **Output buffering during long evals.** When V3 ran for ~3 minutes, Kiro's
   shell output capture timed out before the run finished, producing the
   illusion that the run had crashed. Recovery was to set
   `PYTHONUNBUFFERED=1`, redirect stdout to a file, and read the
   `eval/results_v*.json` files directly from disk after the wait.

**Safety risk specific to this app: budget-violating purchase advice.** If
the LLM recommends "Buy" on a game above the user's stated budget — or invents
a price that doesn't match the catalog — a user could spend money they didn't
plan to. **Mitigation:** every recommendation is checked against
`GAMES_BY_ID` in `app/game_data.py`, prices in the response come from the
catalog rather than from the LLM's freeform text, and
`verify_recommendations` rejects any "Buy" verdict where
`current_price > prefs.budget`. The smart retry only replaces an output that
violates these rules with one that fixes them. **Accepted limit:** the
catalog is a snapshot, not live Steam data. If a price in the JSON is stale,
the verdict will be too. The README and footer disclose this.

---
