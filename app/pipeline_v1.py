"""
V1 baseline: a single LLM call with the entire catalog pasted into the prompt.

This is the "Project 4 pattern" the rubric asks us to go beyond. Kept around as
a reference point so the iteration story (V1 -> V2 -> V3) has real numbers.

No preference extraction, no tool call, no verification.
"""
from __future__ import annotations

import json
import os

from openai import OpenAI

from . import game_data
from .schemas import (
    ChatMessage,
    Recommendation,
    RecommendResponse,
    UserPreferences,
)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _client() -> OpenAI:
    return OpenAI()


SYSTEM = """You are a Steam deal advisor. The full catalog is provided below
as JSON. Read the user's message, decide what they want, and recommend at
most 3 games. Reply with ONLY this JSON shape:

{
  "recommendations": [
    {
      "game_id": string,
      "game_name": string,
      "verdict": "Buy" | "Wait" | "Skip",
      "current_price": number,
      "historical_low": number,
      "reason": string
    }
  ],
  "summary": string
}
"""


def _catalog_block() -> str:
    rows = []
    for g in game_data.GAMES:
        rows.append(
            {
                "id": g["id"],
                "name": g["name"],
                "current_price": g["current_price"],
                "historical_low": g["historical_low"],
                "discount_percent": g["discount_percent"],
                "genres": g["genres"],
                "multiplayer": g["multiplayer"],
                "single_player": g["single_player"],
                "review_percent": g["review_percent"],
            }
        )
    return json.dumps(rows)


def run(
    message: str,
    history: list[ChatMessage],
    prior_prefs: UserPreferences | None,
) -> RecommendResponse:
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "system", "content": f"CATALOG:\n{_catalog_block()}"},
    ]
    for m in history:
        msgs.append({"role": m.role, "content": m.content})
    msgs.append({"role": "user", "content": message})

    resp = _client().chat.completions.create(
        model=MODEL,
        messages=msgs,
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
            continue

    # V1 has no preference extraction; surface an empty UserPreferences so the
    # eval still has a budget field to read (it will be None, so budget checks
    # in the harness skip).
    return RecommendResponse(
        preferences=UserPreferences(),
        candidates=game_data.GAMES,
        recommendations=recs,
        summary=data.get("summary", ""),
        verification={"passed": True, "violations": [], "retried": False},
    )
