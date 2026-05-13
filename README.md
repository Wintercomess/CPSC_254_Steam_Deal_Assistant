# Steam Deal Assistant

An AI-powered Steam deal advisor. Tell it your budget, target price, and the
genres or games you like; it answers with a grounded **Buy / Wait / Skip**
verdict for matching games in a local catalog, with reasoning that cites the
current price, the game's genre, and how it fits your preferences.

The project goes beyond a single LLM call by chaining three stages:

1. **Preference extraction** — the LLM converts freeform chat into a structured
   `UserPreferences` JSON object.
2. **Candidate selection via function calling** — the LLM invokes a local
   `search_games` tool against `data/games.json`. The model picks filters; the
   server runs them deterministically.
3. **Recommendation** — the LLM produces structured Buy/Wait/Skip verdicts
   grounded only in the candidates returned by stage 2.

After stage 3, a **deterministic verification layer** rejects outputs that
recommend games outside the catalog, mark over-budget games as Buy, or omit
price/genre/user-fit reasoning. On a violation, the pipeline retries stage 3
once with the violation message attached.

CPSC 254 Final Project · William Wang.

---

## Requirements

- **Python 3.11+** (the grader environment uses Python 3.11+ on macOS)
- **macOS** (the grader environment) or Windows / Linux for development
- An **OpenAI API key**

No other API keys, no hosted databases, no cloud services. The catalog and
price history are checked-in JSON files in `data/`.

> **Windows note:** if `python --version` shows 3.10 or older, install
> Python 3.11+ from <https://www.python.org/downloads/> and use that
> interpreter (`py -3.11 -m venv .venv`) when creating the virtual environment.

---

## Setup

1. Clone the repo and enter it.

   ```bash
   git clone <this-repo-url>
   cd CPSC-254-Final-Project
   ```

2. Create a virtual environment and install dependencies.

   **macOS / Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   **Windows (PowerShell):**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

   **Windows (cmd):**

   ```cmd
   python -m venv .venv
   .venv\Scripts\activate.bat
   pip install -r requirements.txt
   ```

   > If PowerShell blocks `Activate.ps1` with an execution-policy error, run
   > `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that
   > terminal and try again.

3. Create your `.env` from the template and add your key.

   **macOS / Linux:**

   ```bash
   cp .env.example .env
   ```

   **Windows:**

   ```powershell
   copy .env.example .env
   ```

   Then open `.env` and paste your `OPENAI_API_KEY`.

---

## Run the web app

From the project root, with the venv activated:

```bash
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000> in your browser. Type something like:

> *I have $20 and I love co-op shooters. I already play Apex Legends and Left 4 Dead.*

The app will respond with one or more recommendations, each tagged Buy / Wait /
Skip, and you can ask follow-ups like *"which has more replay value?"* in the
same conversation.

---

## Run the evaluation

The eval harness runs all 12 labeled test cases in `eval/test_cases.json`,
prints a per-case PASS/FAIL line, and writes a timestamped JSON results file to
`eval/`.

```bash
python -m eval.run_eval                  # default = V4 (full pipeline + smart retry)
python -m eval.run_eval --version v1     # baseline: single LLM call
python -m eval.run_eval --version v2     # 3-stage, no verification retry
python -m eval.run_eval --version v3     # 3-stage + naive retry (kept for iteration history)
python -m eval.run_eval --version v4     # 3-stage + smart retry-with-fallback
```

Four pipeline versions are kept side-by-side so the iteration story in
`REPORT.md` is reproducible:

| Version | What changes |
|---|---|
| `app/pipeline_v1.py` | Single LLM call, full catalog pasted into the prompt. No tools, no verification. |
| `app/pipeline_v2.py` | Three stages: extraction → tool-calling search → recommendation. No retry on verification failure. |
| `app/pipeline_v3.py` | V2 plus a naive verification retry that unconditionally replaces the original output. |
| `app/pipeline.py` (V4) | V3 with a smarter retry: only adopt the retry when it has at least one rec and reduces violation count. The web app uses this version. |

The top-level metric is **recommendation_pass_rate**:

```
recommendation_pass_rate =
    (# test cases where the AI recommends only catalog games,
     respects the user's budget on Buy verdicts,
     uses Buy/Wait/Skip,
     mentions price + genre + user fit in every reason,
     and matches the per-case domain expectations)
    / (# total test cases)
```

Sub-metrics (`valid_game_rate`, `budget_compliance_rate`,
`verdict_format_rate`, `explanation_completeness`, `expectation_match_rate`)
are also reported separately so iteration deltas are easy to read.

---

## Project layout

```
.
├── app/
│   ├── main.py          # FastAPI app + /api/recommend route
│   ├── pipeline.py      # V4: 3-stage LLM pipeline + smart retry-with-fallback
│   ├── pipeline_v1.py   # V1 baseline: single LLM call (kept for eval)
│   ├── pipeline_v2.py   # V2: 3-stage, no verification (kept for eval)
│   ├── pipeline_v3.py   # V3: 3-stage + naive retry (kept for eval)
│   ├── game_data.py     # Local catalog loader + search_games tool
│   └── schemas.py       # Pydantic request/response models
├── data/
│   ├── games.json       # 20 sample Steam games
│   └── price_history.json
├── frontend/
│   ├── index.html       # Single-page UI
│   ├── app.js           # Chat client, multi-turn state
│   └── styles.css
├── eval/
│   ├── test_cases.json  # 12 labeled test cases
│   └── run_eval.py      # Eval harness, writes results_*.json
├── requirements.txt
├── .env.example
├── REPORT.md
└── README.md
```

---

## Notes for graders

- The grader-supplied `.env` only needs `OPENAI_API_KEY`. Nothing else.
- All catalog data lives in `data/` and is committed to the repo.
- The default model is `gpt-4o-mini`. To override during eval, set
  `OPENAI_MODEL` in your environment before running `python -m eval.run_eval`.
