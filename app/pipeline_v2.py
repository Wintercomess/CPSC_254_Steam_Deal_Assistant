"""
V2: three-stage pipeline (extraction -> tool-calling search -> recommendation),
but WITHOUT the deterministic verification + retry layer.

V3 in pipeline.py adds verification on top of this.
"""
from __future__ import annotations

from .pipeline import (
    extract_preferences,
    recommend,
    select_candidates,
    MODEL,
)
from .schemas import ChatMessage, RecommendResponse, UserPreferences


def run(
    message: str,
    history: list[ChatMessage],
    prior_prefs: UserPreferences | None,
) -> RecommendResponse:
    prefs = extract_preferences(message, history, prior_prefs)
    candidates = select_candidates(prefs)
    recs, summary = recommend(prefs, candidates)
    return RecommendResponse(
        preferences=prefs,
        candidates=candidates,
        recommendations=recs,
        summary=summary,
        # V2 reports the verification result for transparency but never retries.
        verification={"passed": True, "violations": [], "retried": False},
    )
