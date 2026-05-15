# REPORT — Steam Deal Scout AI

CPSC 254 Final Project · William Wang · gpt-4o-mini · 12 labeled test cases

---

## 1. What & why

Steam Deal Scout AI is a web app that helps PC gamers decide whether to **Buy**, **Wait**, or **Skip** a discounted game. The user types a message such as, “I have $20 and I love co-op shooters,” and the app returns one to three recommendations from a local Steam-style catalog. Each recommendation includes the game name, current price, historical low, verdict, and a short explanation that connects the deal to the user’s budget, genre interests, and playstyle.

The target user is myself and friends who buy games during Steam sales but do not always know whether a discount is actually worth it. A normal deal tracker can show a low price, but it does not explain whether the game fits the user personally.

The hard AI problem is grounding. If the model is allowed to answer freely, it may recommend games that are not in the dataset, invent prices, or mark an over-budget game as “Buy” because it seems like a good title. A single text-in → structured-output call is not enough because there is no reliable source of truth. This project uses a local game catalog, tool-based candidate search, structured recommendation, and deterministic verification so the final answer is more controlled and testable.

---

## 2. Iterations

All versions were measured on the same 12 labeled test cases in `eval/test_cases.json` using `recommendation_pass_rate`.

### V1 — Single prompt baseline

**Change:** V1 used one LLM call with the user message and the whole catalog included in the prompt. There was no preference extraction, tool call, or verification layer.

**Motivating example:** In `tc09_overbudget_should_not_buy`, the user had a $30 budget and asked for a souls-like game. The model recommended Elden Ring at $41.99 and also invented a non-catalog game.

**Delta:** Baseline score was **8/12 = 66.7%**.

**Conclusion:** The score was acceptable on easy cases, but the failure showed that prompt instructions alone did not stop hallucinated games or bad budget decisions. The next version needed a grounded candidate-selection step.

### V2 — Function-calling candidate search

**Change:** V2 split the system into stages. First, the model extracted structured user preferences. Then it called a local `search_games` tool that filtered `games.json`. Finally, the recommendation step could only choose from the returned candidates.

**Motivating example:** V1’s hallucinated recommendation in `tc09` motivated this change. After adding the local search tool, every recommended game came from the catalog.

**Delta:** Overall score stayed at **8/12 = 66.7%**, but valid-game accuracy improved from **83.3% to 100%**.

**Conclusion:** The tool call solved hallucinated games, but it created a new issue. Sometimes the model chose filters that were too strict, so the best game was excluded before the recommendation step. The next version needed verification and retry logic.

### V3 — Verification with naive retry

**Change:** V3 added `verify_recommendations`, which checks for non-catalog games, over-budget Buy verdicts, and missing explanation evidence. If the verifier found a problem, the app retried the recommendation once.

**Motivating example:** In `tc04_singleplayer_only`, the first answer was mostly useful, but the verifier flagged missing wording. The retry returned an empty recommendation list, and V3 replaced the original answer with that worse result.

**Delta:** Score dropped from **8/12 to 7/12**, or **66.7% → 58.3%**.

**Conclusion:** The verification idea was good, but the retry policy was too aggressive. Replacing the original answer with an empty retry made the system less reliable. The next version needed a fallback rule.

### V4 — Smart retry with fallback

**Change:** V4 kept the verifier, but only accepts the retry if it returns at least one recommendation and reduces the number of violations. If the retry is worse, the app keeps the original answer and reports the verification result.

**Motivating example:** Replaying `tc04_singleplayer_only`, the empty retry is now rejected, and the original useful recommendations remain visible.

**Delta:** Score recovered from **7/12 to 8/12**, or **58.3% → 66.7%**. Per-call validity checks returned to 100%, while some expectation-matching cases still failed.

**Conclusion:** The fallback made the verifier safer. The remaining failures mostly happen when strict search filters hide a relevant game. Next, I would add a relaxed fallback search or force-include games that the user names directly.

---

## 3. Code walkthrough

When a user submits a message, `frontend/app.js:130` handles the form event, prevents the default page reload, adds the user bubble, and calls `send(msg)`. Inside `send`, `frontend/app.js:91-99` sends a JSON POST request to `/api/recommend` with the current message, chat history, and saved preferences. It also shows a loading status and catches server errors in `frontend/app.js:121-127`.

The backend route starts in `app/main.py:20-27`. FastAPI validates the request with `RecommendRequest` from `app/schemas.py:19-23`, rejects empty messages, and then calls `pipeline.run`. The main pipeline is in `app/pipeline.py:384-420`. First, `extract_preferences` in `app/pipeline.py:73-109` converts the natural-language message into structured fields such as budget, target price, genres, liked games, and playstyle. Next, `select_candidates` in `app/pipeline.py:156-226` lets the model call the local `search_games` tool. The tool itself is defined in `app/pipeline.py:115-148`, but the actual deterministic filtering happens in `app/game_data.py:32-66` using the local game catalog.

After candidate selection, `recommend` in `app/pipeline.py:282-320` asks the model to produce Buy/Wait/Skip recommendations using only the candidate games. Then `verify_recommendations` in `app/pipeline.py:326-378` checks catalog membership, budget rules, and explanation quality. If verification fails, `app/pipeline.py:395-412` retries once and only keeps the retry if it improves the result.

One design decision was using function calling instead of putting the whole catalog into one prompt. I rejected the simpler full-catalog prompt because V1 showed that the model could still hallucinate titles or prices. Function calling made the catalog search more controlled and gave the verifier a smaller, exact candidate set to check.

---

## 4. AI disclosure & safety

I used **Codex in VS Code** as my coding assistant to generate the first project structure, refine the pipeline, and help debug errors. One failure was that Codex initially pushed the project toward a larger design than I needed. I corrected this by keeping the app local and avoiding extra services, since the grader only supplies an OpenAI API key. Another failure was the naive V3 retry logic, where Codex helped add a retry but the eval showed that it could replace a usable answer with an empty one. I fixed that by adding the V4 fallback rule. A third issue was setup confusion around Python and `uvicorn`; I tested the commands manually in the virtual environment and adjusted the README so the run command matched the actual FastAPI app.

A safety risk specific to this app is misleading purchase advice. If the model recommends “Buy” for a game above the user’s budget or uses an invented price, the user could make a poor spending decision. To reduce that risk, the app only recommends catalog games, uses saved prices from local data, verifies Buy verdicts against the budget, and labels the result as a recommendation rather than a guaranteed best deal. The accepted limit is that the catalog is not live Steam data, so prices may be stale unless the dataset is updated.
