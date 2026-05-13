"""
Eval harness for the Steam Deal Assistant.

Metric (top level):
    recommendation_pass_rate =
        (# test cases where ALL sub-checks pass) / (# total test cases)

Sub-metrics (each reported separately for honest debugging):
    - valid_game_rate           : recs only contain games from the catalog
    - budget_compliance_rate    : no "Buy" verdict above the user's budget
    - verdict_format_rate       : every rec has Buy/Wait/Skip
    - explanation_completeness  : every rec mentions price + genre + user fit
    - expectation_match_rate    : per-test-case domain checks (see test_cases.json)

Run:
    python -m eval.run_eval               # default = V4 (full pipeline + smart retry)
    python -m eval.run_eval --version v1
    python -m eval.run_eval --version v2
    python -m eval.run_eval --version v3
    python -m eval.run_eval --version v4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make the repo root importable when run as a module or as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv()

from app import pipeline, pipeline_v1, pipeline_v2, pipeline_v3  # noqa: E402
from app.schemas import ChatMessage, UserPreferences  # noqa: E402

PIPELINES = {
    "v1": pipeline_v1,
    "v2": pipeline_v2,
    "v3": pipeline_v3,
    "v4": pipeline,
}

TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"


def _evaluate_case(case: dict, response) -> dict:
    """Return a dict of pass/fail flags for one test case."""
    expect = case.get("expect", {})
    recs = response.recommendations
    rec_ids = [r.game_id for r in recs]

    flags: dict[str, bool] = {}
    notes: list[str] = []

    # Sub-metric 1: every game must be in the catalog (verification already
    # checks this, but we double-check for the eval report).
    from app.game_data import GAMES_BY_ID
    flags["valid_game"] = all(rid in GAMES_BY_ID for rid in rec_ids) and len(rec_ids) > 0

    # Sub-metric 2: budget compliance for Buy verdicts
    budget = response.preferences.budget
    if budget is None:
        flags["budget_compliance"] = True
    else:
        flags["budget_compliance"] = all(
            r.verdict != "Buy" or r.current_price <= budget for r in recs
        )

    # Sub-metric 3: verdict format
    flags["verdict_format"] = all(r.verdict in ("Buy", "Wait", "Skip") for r in recs)

    # Sub-metric 4: explanation completeness (server-side verification result)
    flags["explanation_complete"] = response.verification.get("passed", False) or (
        # If a retry succeeded, the final state is still "passed"; if not, fail.
        len(response.verification.get("violations", [])) == 0
    )

    # Sub-metric 5: per-test-case expectations
    case_ok = True

    if "must_recommend_any_of" in expect:
        if not any(rid in expect["must_recommend_any_of"] for rid in rec_ids):
            case_ok = False
            notes.append(f"none of {expect['must_recommend_any_of']} in {rec_ids}")

    if "must_not_recommend" in expect:
        bad = [rid for rid in rec_ids if rid in expect["must_not_recommend"]]
        if bad:
            case_ok = False
            notes.append(f"recommended forbidden: {bad}")

    if "max_price_for_buy" in expect:
        cap = expect["max_price_for_buy"]
        bad = [r.game_name for r in recs if r.verdict == "Buy" and r.current_price > cap]
        if bad:
            case_ok = False
            notes.append(f"Buy verdict over ${cap}: {bad}")

    if "must_not_recommend_buy_above" in expect:
        cap = expect["must_not_recommend_buy_above"]
        bad = [r.game_name for r in recs if r.verdict == "Buy" and r.current_price > cap]
        if bad:
            case_ok = False
            notes.append(f"Buy verdict over ${cap}: {bad}")

    if "expected_verdict_for" in expect:
        for gid, want in expect["expected_verdict_for"].items():
            actual = next((r.verdict for r in recs if r.game_id == gid), None)
            if actual is not None and actual != want:
                case_ok = False
                notes.append(f"{gid}: expected {want}, got {actual}")

    if "must_mention_keywords" in expect:
        text = " ".join(r.reason.lower() for r in recs)
        missing = [kw for kw in expect["must_mention_keywords"] if kw.lower() not in text]
        if missing:
            case_ok = False
            notes.append(f"missing keywords: {missing}")

    if expect.get("must_only_recommend_from_catalog"):
        if not flags["valid_game"]:
            case_ok = False
            notes.append("at least one rec was not in catalog")

    flags["expectation_match"] = case_ok

    overall = all(flags.values())
    return {
        "id": case["id"],
        "passed": overall,
        "flags": flags,
        "notes": notes,
        "rec_ids": rec_ids,
        "verdicts": [r.verdict for r in recs],
        "verification": response.verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval against a pipeline version.")
    parser.add_argument(
        "--version",
        choices=list(PIPELINES.keys()),
        default="v4",
        help="Which pipeline to evaluate (default: v4 = full pipeline + smart retry).",
    )
    args = parser.parse_args()
    selected = PIPELINES[args.version]

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Add it to .env.", file=sys.stderr)
        return 1

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"Running {len(cases)} eval cases against pipeline={args.version} model={selected.MODEL}\n")

    results: list[dict] = []
    started = time.time()
    for i, case in enumerate(cases, 1):
        history = [ChatMessage(**m) for m in case.get("history", [])]
        prior = UserPreferences(**case["preferences"]) if case.get("preferences") else None
        try:
            resp = selected.run(case["message"], history, prior)
            outcome = _evaluate_case(case, resp)
        except Exception as exc:
            outcome = {
                "id": case["id"],
                "passed": False,
                "flags": {},
                "notes": [f"pipeline error: {exc}"],
                "rec_ids": [],
                "verdicts": [],
                "verification": {"passed": False, "violations": [str(exc)]},
            }
        results.append(outcome)
        mark = "PASS" if outcome["passed"] else "FAIL"
        print(f"  [{i:02d}/{len(cases)}] {mark}  {case['id']}")
        if not outcome["passed"]:
            for n in outcome["notes"]:
                print(f"        - {n}")

    elapsed = time.time() - started

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    sub = {
        key: sum(1 for r in results if r["flags"].get(key, False)) / total
        for key in (
            "valid_game",
            "budget_compliance",
            "verdict_format",
            "explanation_complete",
            "expectation_match",
        )
    }

    print("\n=== Summary ===")
    print(f"recommendation_pass_rate : {passed}/{total} = {passed/total:.2%}")
    for k, v in sub.items():
        print(f"{k:<26}: {v:.2%}")
    print(f"elapsed                  : {elapsed:.1f}s")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / f"results_{args.version}_{stamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": args.version,
                "model": selected.MODEL,
                "total": total,
                "passed": passed,
                "pass_rate": passed / total,
                "sub_metrics": sub,
                "elapsed_seconds": elapsed,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
