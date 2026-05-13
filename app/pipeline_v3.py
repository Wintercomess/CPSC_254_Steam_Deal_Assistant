"""
V3 (frozen): three-stage pipeline + verification with a naive retry that
unconditionally replaces the original output, even when the retry produces an
empty recommendation list.

Kept around so the V3 -> V4 delta in REPORT.md stays reproducible. V4 lives in
pipeline.py and is what the web app serves.
"""
from __future__ import annotations

from .pipeline import (
    MODEL,
    extract_preferences,
    recommend,
    select_candidates,
    verify_recommendations,
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
    passed, violations = verify_recommendations(prefs, candidates, recs)

    retried = False
    if not passed:
        retried = True
        # NAIVE retry: replace blindly, no fallback. This is the V3 bug.
        recs, summary = recommend(prefs, candidates, extra_violation_note="; ".join(violations))
        passed, violations = verify_recommendations(prefs, candidates, recs)

    return RecommendResponse(
        preferences=prefs,
        candidates=candidates,
        recommendations=recs,
        summary=summary,
        verification={"passed": passed, "violations": violations, "retried": retried},
    )
