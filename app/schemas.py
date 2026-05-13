"""Pydantic models for the public HTTP API."""
from typing import Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class UserPreferences(BaseModel):
    budget: float | None = None
    target_price: float | None = None
    genres: list[str] = Field(default_factory=list)
    owned_or_liked_games: list[str] = Field(default_factory=list)
    playstyle: str | None = None  # e.g. "single-player", "co-op"


class RecommendRequest(BaseModel):
    """A single user message + optional prior turns + optional structured prefs."""
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    preferences: UserPreferences | None = None


class Recommendation(BaseModel):
    game_id: str
    game_name: str
    verdict: Literal["Buy", "Wait", "Skip"]
    current_price: float
    historical_low: float
    reason: str


class RecommendResponse(BaseModel):
    preferences: UserPreferences
    candidates: list[dict]  # raw catalog rows the LLM was allowed to choose from
    recommendations: list[Recommendation]
    summary: str
    verification: dict  # {"passed": bool, "violations": [...], "retried": bool}
