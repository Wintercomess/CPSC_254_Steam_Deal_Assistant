"""FastAPI entrypoint. Serves the static frontend and the /api/recommend route."""
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()  # read OPENAI_API_KEY from .env

from . import pipeline  # noqa: E402  (import after load_dotenv)
from .schemas import RecommendRequest, RecommendResponse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="Steam Deal Assistant", version="1.0.0")


@app.post("/api/recommend", response_model=RecommendResponse)
def recommend_endpoint(req: RecommendRequest) -> RecommendResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")
    try:
        return pipeline.run(req.message, req.history, req.preferences)
    except Exception as exc:  # surface a clean error to the UI
        raise HTTPException(status_code=500, detail=f"pipeline failure: {exc}") from exc


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


# Static frontend (index.html + app.js + styles.css)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
