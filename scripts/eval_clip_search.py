#!/usr/bin/env python3
import argparse
import json
import sys
from urllib import request, error


def _read_cases(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    raise ValueError("Cases file must be a list or an object with key 'cases'.")


def _post_json(url, data, headers):
    body = json.dumps(data).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=120) as res:
        raw = res.read().decode("utf-8")
        return res.status, json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Clip Anything search quality against labeled queries.")
    parser.add_argument("--api-base", default="http://localhost:8000", help="API base URL (default: http://localhost:8000)")
    parser.add_argument("--job-id", required=True, help="Job ID to evaluate")
    parser.add_argument("--cases", required=True, help="JSON file with evaluation cases")
    parser.add_argument("--search-mode", default="balanced", choices=["exact", "balanced", "broad"], help="Search mode")
    parser.add_argument("--limit", type=int, default=6, help="Per-query match limit")
    parser.add_argument("--shortlist-limit", type=int, default=6, help="Per-query shortlist limit")
    parser.add_argument("--overlap-threshold", type=float, default=0.35, help="Expected overlap threshold")
    parser.add_argument("--gemini-key", default=None, help="Optional Gemini key header")
    args = parser.parse_args()

    cases = _read_cases(args.cases)
    payload = {
        "job_id": args.job_id,
        "cases": cases,
        "search_mode": args.search_mode,
        "limit": max(1, min(20, int(args.limit))),
        "shortlist_limit": max(1, min(12, int(args.shortlist_limit))),
        "expected_overlap_threshold": max(0.05, min(0.95, float(args.overlap_threshold))),
    }
    headers = {"Content-Type": "application/json"}
    if args.gemini_key:
        headers["X-Gemini-Key"] = args.gemini_key

    url = f"{args.api_base.rstrip('/')}/api/search/clips/eval"
    try:
        status, data = _post_json(url, payload, headers)
    except error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        print(f"HTTP {e.code}: {detail}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    if status < 200 or status >= 300:
        print(f"Unexpected status: {status}", file=sys.stderr)
        return 1

    print("\n=== Clip Search Eval ===")
    print(f"job_id: {data.get('job_id')}")
    print(f"cases: {data.get('total_cases')} | passed: {data.get('passed_cases')} | pass_rate: {data.get('pass_rate')}")
    print(f"mean_top_match_score: {data.get('mean_top_match_score')} | mean_top_overlap: {data.get('mean_top_overlap')} | mrr: {data.get('mrr')}")
    print(f"search_mode: {data.get('search_mode')} | overlap_threshold: {data.get('overlap_threshold')}")

    print("\nPer-case:")
    for item in data.get("details", []):
        idx = item.get("case_index")
        query = item.get("query", "")
        passed = "PASS" if item.get("passed") else "FAIL"
        top_score = item.get("top_match_score")
        top_overlap = item.get("top_overlap")
        first_hit = item.get("first_hit_rank")
        err = item.get("error")
        if err:
            print(f"[{idx}] {passed} | {query} | error={err}")
            continue
        print(
            f"[{idx}] {passed} | score={top_score} | overlap={top_overlap} | "
            f"first_hit_rank={first_hit} | {query}"
        )
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
