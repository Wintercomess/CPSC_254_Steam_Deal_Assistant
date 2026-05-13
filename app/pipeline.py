"""
Three-stage AI pipeline for the Steam Deal Assistant.

Stage A: extract_preferences  -- natural language -> structured UserPreferences
Stage B: select_candidates    -- LLM uses tool/function calling to query the
                                 local catalog deterministically
Stage C: recommend            -- LLM produces Buy/Wait/Skip + reasoning grounded
                                 ONLY in the candidates returned by stage B

After stage C, verify_recommendations() runs a deterministic post-check. If a
violation is found (hallucinated game, over-budget pick, missing fields), the
pipeline retries stage C exactly once with the violation message appended.
The retry result is only adopted when it has at least one recommendation AND
fixes (or reduces) the violation count; otherwise the original response is
kept. This avoids the V3 failure mode where a blind retry produced an empty
list and replaced a working answer.

This multi-stage + tool + verify shape is what makes the project go beyond a
single extraction call.
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from . import game_data
from .schemas import (
    ChatMessage,
    Recommendation,
    RecommendResponse,
    UserPreferences,
)

# ---- Model selection -------------------------------------------------------
# Using gpt-4o-mini as the default. It is cheap, fast, and supports tools.
# The eval can override via env var so V1/V2/V3 can compare models if needed.
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _client() -> OpenAI:
    """Lazy client so importing this module doesn't require an API key."""
    return OpenAI()


# ---------------------------------------------------------------------------
# Stage A: preference extraction
# ---------------------------------------------------------------------------
PREF_EXTRACTION_SYSTEM = """You extract a Steam shopper's preferences from
their message and prior chat history. Return ONLY valid JSON with this exact
shape:

{
  "budget": number or null,
  "target_price": number or null,
  "genres": [string, ...],
  "owned_or_liked_games": [string, ...],
  "playstyle": "single-player" | "co-op" | "multiplayer" | "any" | null
}

Rules:
- "budget" is the most the user is willing to spend on one game right now.
- "target_price" is a stricter price they would prefer to wait for. Null if
  not mentioned.
- Genres should be lowercase keywords like "fps", "rpg", "co-op", "roguelike",
  "indie", "open-world", "survival", "metroidvania".
- If a value is not stated or implied, use null or [].
- Do NOT invent fields. Do NOT add commentary."""


def extract_preferences(
    message: str,
    history: list[ChatMessage],
    prior: UserPreferences | None,
) -> UserPreferences:
    """Stage A: turn freeform chat into a structured UserPreferences."""
    msgs: list[dict[str, str]] = [{"role": "system", "content": PREF_EXTRACTION_SYSTEM}]
    if prior is not None:
        msgs.append(
            {
                "role": "system",
                "content": f"Existing preferences (merge, do not overwrite with null): {prior.model_dump_json()}",
            }
        )
    for m in history:
        msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": message})

    resp = _client().chat.completions.create(
        model=MODEL,
        messages=msgs,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    merged: dict[str, Any] = prior.model_dump() if prior else {}
    for key in ("budget", "target_price", "genres", "owned_or_liked_games", "playstyle"):
        val = data.get(key)
        if val in (None, [], ""):
            continue
        merged[key] = val
    return UserPreferences(**merged)


# ---------------------------------------------------------------------------
# Stage B: candidate selection via function/tool calling
# ---------------------------------------------------------------------------
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_games",
        "description": (
            "Search the local Steam catalog for games matching the user's "
            "filters. Returns at most 8 games. Always call this before making "
            "a recommendation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Genre keywords like fps, rpg, co-op, roguelike.",
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum current price in USD.",
                },
                "multiplayer": {
                    "type": "boolean",
                    "description": "True to require multiplayer; False to require single-player; omit for either.",
                },
                "min_review_percent": {
                    "type": "integer",
                    "description": "Minimum positive-review percentage (0-100).",
                },
            },
            "required": [],
        },
    },
}

CANDIDATE_SYSTEM = """You are a Steam deal advisor. The user's preferences are
provided as JSON. Call the `search_games` tool one or more times to gather the
games you will consider. After you have enough candidates, reply with the
literal string DONE and nothing else."""


def select_candidates(prefs: UserPreferences) -> list[dict[str, Any]]:
    """Stage B: let the model pick search filters; we run them locally."""
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": CANDIDATE_SYSTEM},
        {"role": "user", "content": f"User preferences: {prefs.model_dump_json()}"},
    ]

    seen: dict[str, dict[str, Any]] = {}
    # Hard cap on tool-call rounds to prevent cost runaway.
    for _ in range(3):
        resp = _client().chat.completions.create(
            model=MODEL,
            messages=msgs,
            tools=[SEARCH_TOOL],
            temperature=0.0,
        )
        choice = resp.choices[0].message
        if not choice.tool_calls:
            break

        # Append the assistant's tool-call turn (required by the API contract).
        msgs.append(
            {
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.tool_calls
                ],
            }
        )

        for tc in choice.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            results = game_data.search_games(
                genres=args.get("genres"),
                max_price=args.get("max_price"),
                multiplayer=args.get("multiplayer"),
                min_review_percent=args.get("min_review_percent"),
            )
            for r in results:
                seen[r["id"]] = r
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(results),
                }
            )

    # Fallback: if the model declined to call the tool at all, run a budget-only
    # search so stage C still has something grounded to work with.
    if not seen:
        results = game_data.search_games(
            genres=prefs.genres or None,
            max_price=prefs.budget,
        )
        for r in results:
            seen[r["id"]] = r

    return list(seen.values())


# ---------------------------------------------------------------------------
# Stage C: recommendation with grounded reasoning
# ---------------------------------------------------------------------------
RECOMMEND_SYSTEM = """You are a Steam deal advisor. You will receive:
- the user's preferences as JSON
- a list of candidate games (the ONLY games you may recommend)
- price history for each candidate

Return ONLY valid JSON with this exact shape:

{
  "recommendations": [
    {
      "game_id": string,        // must be one of the candidate ids
      "game_name": string,
      "verdict": "Buy" | "Wait" | "Skip",
      "current_price": number,
      "historical_low": number,
      "reason": string          // 1-3 sentences. Must mention price, genre, and how it fits the user's preference.
    }
  ],
  "summary": string             // a short overall takeaway, 1-2 sentences
}

Rules:
- Recommend AT MOST 3 games, ranked best fit first.
- Verdict "Buy" only if current_price <= user budget AND game matches at least one user genre/playstyle.
- Verdict "Wait" if current_price is above the user's target_price OR notably above the historical_low.
- Verdict "Skip" if the game doesn't fit the user's preferences.
- NEVER recommend a game that is not in the candidate list.
- Every reason MUST mention the price, the genre, and why it fits (or doesn't fit) the user."""


def _format_candidates(candidates: list[dict[str, Any]]) -> str:
    rows = []
    for c in candidates:
        rows.append(
            {
                "id": c["id"],
                "name": c["name"],
                "current_price": c["current_price"],
                "historical_low": c["historical_low"],
                "discount_percent": c["discount_percent"],
                "genres": c["genres"],
                "multiplayer": c["multiplayer"],
                "single_player": c["single_player"],
                "review_percent": c["review_percent"],
                "price_history": game_data.get_price_history(c["id"]),
            }
        )
    return json.dumps(rows, indent=2)


def recommend(
    prefs: UserPreferences,
    candidates: list[dict[str, Any]],
    extra_violation_note: str | None = None,
) -> tuple[list[Recommendation], str]:
    """Stage C: produce structured recommendations from the candidate set."""
    user_block = (
        f"User preferences:\n{prefs.model_dump_json(indent=2)}\n\n"
        f"Candidates (the only allowed games):\n{_format_candidates(candidates)}"
    )
    if extra_violation_note:
        user_block += (
            "\n\nIMPORTANT: your previous response failed verification: "
            f"{extra_violation_note}\nFix it and respond again."
        )

    resp = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RECOMMEND_SYSTEM},
            {"role": "user", "content": user_block},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"recommendations": [], "summary": ""}

    recs: list[Recommendation] = []
    for item in data.get("recommendations", []):
        try:
            recs.append(Recommendation(**item))
        except Exception:
            # Drop malformed items; verification will flag the shortfall.
            continue
    return recs, data.get("summary", "")


# ---------------------------------------------------------------------------
# Verification (deterministic post-check)
# ---------------------------------------------------------------------------
def verify_recommendations(
    prefs: UserPreferences,
    candidates: list[dict[str, Any]],
    recs: list[Recommendation],
) -> tuple[bool, list[str]]:
    """Return (passed, list_of_violations). Empty violations == passed."""
    violations: list[str] = []
    candidate_ids = {c["id"] for c in candidates}

    if not recs:
        violations.append("no recommendations returned")

    for r in recs:
        if r.game_id not in candidate_ids:
            violations.append(
                f"hallucinated game '{r.game_id}' not in candidate set"
            )
            continue

        catalog = game_data.get_game(r.game_id)
        if catalog is None:
            violations.append(f"game '{r.game_id}' missing from catalog")
            continue

        # Buy verdict must respect budget
        if r.verdict == "Buy" and prefs.budget is not None:
            if catalog["current_price"] > prefs.budget:
                violations.append(
                    f"'{r.game_name}' marked Buy at ${catalog['current_price']} "
                    f"but user budget is ${prefs.budget}"
                )

        # Reason must reference price, genre, and user fit signals
        reason = r.reason.lower()
        has_price = any(tok in reason for tok in ("$", "price", "cost", "discount", "sale"))
        has_genre = any(g.lower() in reason for g in catalog.get("genres", []))
        has_fit = any(
            tok in reason
            for tok in ("you", "your", "prefer", "match", "fit", "budget", "target")
        )
        if not (has_price and has_genre and has_fit):
            missing = []
            if not has_price:
                missing.append("price")
            if not has_genre:
                missing.append("genre")
            if not has_fit:
                missing.append("user fit")
            violations.append(
                f"'{r.game_name}' reason missing: {', '.join(missing)}"
            )

    return (len(violations) == 0, violations)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def run(
    message: str,
    history: list[ChatMessage],
    prior_prefs: UserPreferences | None,
) -> RecommendResponse:
    prefs = extract_preferences(message, history, prior_prefs)
    candidates = select_candidates(prefs)
    recs, summary = recommend(prefs, candidates)
    passed, violations = verify_recommendations(prefs, candidates, recs)

    retried = False
    if not passed:
        retried = True
        retry_recs, retry_summary = recommend(
            prefs, candidates, extra_violation_note="; ".join(violations)
        )
        retry_passed, retry_violations = verify_recommendations(
            prefs, candidates, retry_recs
        )
        # V4 fix: only adopt the retry if it actually produced recommendations
        # AND it didn't make things worse. Empty retry replaces are the V3 bug.
        retry_is_better = (
            len(retry_recs) > 0
            and (retry_passed or len(retry_violations) < len(violations))
        )
        if retry_is_better:
            recs, summary = retry_recs, retry_summary
            passed, violations = retry_passed, retry_violations
        # else: keep the original output and report that the retry didn't help.

    return RecommendResponse(
        preferences=prefs,
        candidates=candidates,
        recommendations=recs,
        summary=summary,
        verification={"passed": passed, "violations": violations, "retried": retried},
    )
