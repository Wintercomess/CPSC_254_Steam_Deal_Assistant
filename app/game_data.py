"""
Local game catalog and price history loader.

This module is the single source of truth about what games exist. The LLM is
NEVER allowed to invent a game; the verification layer in pipeline.py checks
every recommended game id against `GAMES_BY_ID` here.
"""
import json
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GAMES_PATH = DATA_DIR / "games.json"
HISTORY_PATH = DATA_DIR / "price_history.json"


def _load_games() -> list[dict[str, Any]]:
    with open(GAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_history() -> dict[str, list[dict[str, Any]]]:
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


GAMES: list[dict[str, Any]] = _load_games()
GAMES_BY_ID: dict[str, dict[str, Any]] = {g["id"]: g for g in GAMES}
PRICE_HISTORY: dict[str, list[dict[str, Any]]] = _load_history()


def search_games(
    genres: list[str] | None = None,
    max_price: float | None = None,
    multiplayer: bool | None = None,
    min_review_percent: int | None = None,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """
    Deterministic catalog filter. Called by the LLM via function calling.

    Returns up to `limit` games matching all provided filters. Sort order:
    higher review_percent first, then larger discount_percent.
    """
    results: list[dict[str, Any]] = []
    for game in GAMES:
        if max_price is not None and game["current_price"] > max_price:
            continue
        if multiplayer is True and not game.get("multiplayer", False):
            continue
        if multiplayer is False and not game.get("single_player", False):
            continue
        if min_review_percent is not None and game["review_percent"] < min_review_percent:
            continue
        if genres:
            wanted = {g.lower() for g in genres}
            have = {g.lower() for g in game.get("genres", [])}
            if not (wanted & have):
                continue
        results.append(game)

    results.sort(
        key=lambda g: (g["review_percent"], g["discount_percent"]),
        reverse=True,
    )
    return results[:limit]


def get_price_history(game_id: str) -> list[dict[str, Any]]:
    """Return the saved price snapshots for a given game id, or []."""
    return PRICE_HISTORY.get(game_id, [])


def get_game(game_id: str) -> dict[str, Any] | None:
    return GAMES_BY_ID.get(game_id)
