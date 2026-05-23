"""
run_golden_set.py
-----------------
Fires all 70 golden set queries at the n8n webhook in sequence.
Adds deliberate delays between requests to avoid Groq + Gemini rate limits.
The n8n workflow logs every execution to Airtable automatically.
This script also saves a local backup CSV of all webhook responses.

Usage:
    python run_golden_set.py

Prerequisites:
    - n8n workflow "Intelligent AI Router" is active
    - .env file populated (see .env.example)
    - pip install -r requirements.txt
"""

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
WEBHOOK_URL     = os.environ["N8N_WEBHOOK_URL"]
GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.csv"
BACKUP_CSV_PATH = Path(__file__).parent / "golden_set_responses.csv"
DELAY_SECONDS   = 3      # Prevents Groq + Gemini free-tier rate limit hits
REQUEST_TIMEOUT = 45     # Seconds — generous for cold starts

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_golden_set(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fire_query(query: dict) -> dict:
    """POST a single query to the webhook. Returns the response JSON."""
    payload = {
        "request_id":      query["query_id"],   # Use query_id as request_id for traceability
        "user_query":      query["user_query"],
        "expected_intent": query["expected_intent"],
    }
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        result = resp.json()
        result["_query_id"]       = query["query_id"]
        result["_query_category"] = query["query_category"]
        result["_http_status"]    = resp.status_code
        return result
    except requests.exceptions.Timeout:
        print(f"    ✗ TIMEOUT after {REQUEST_TIMEOUT}s")
        return {
            "_query_id":       query["query_id"],
            "_query_category": query["query_category"],
            "_http_status":    None,
            "error":           "TIMEOUT",
            "request_id":      query["query_id"],
            "user_query":      query["user_query"],
            "expected_intent": query["expected_intent"],
        }
    except Exception as e:
        print(f"    ✗ ERROR: {e}")
        return {
            "_query_id":       query["query_id"],
            "_query_category": query["query_category"],
            "_http_status":    None,
            "error":           str(e),
            "request_id":      query["query_id"],
            "user_query":      query["user_query"],
            "expected_intent": query["expected_intent"],
        }


def save_backup(results: list[dict], path: Path) -> None:
    """Write all results to a local CSV backup."""
    if not results:
        return
    fieldnames = sorted({k for r in results for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"\nLocal backup saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    queries = load_golden_set(GOLDEN_SET_PATH)
    total   = len(queries)
    print(f"Loaded {total} queries from golden_set.csv")
    print(f"Webhook: {WEBHOOK_URL}")
    print(f"Delay between requests: {DELAY_SECONDS}s")
    print(f"Started at: {datetime.now().isoformat()}")
    print("─" * 60)

    results  = []
    errors   = 0
    successes = 0

    for i, query in enumerate(queries, 1):
        qid      = query["query_id"]
        category = query["query_category"]
        expected = query["expected_intent"]

        print(f"[{i:02d}/{total}] {qid} ({category}) → expected: {expected}")
        result = fire_query(query)

        actual   = result.get("actual_intent", "ERROR")
        path     = result.get("routing_path", "?")
        latency  = result.get("total_latency_ms", "?")
        is_error = "error" in result

        if is_error:
            errors += 1
            print(f"    ✗ FAILED: {result.get('error')}")
        else:
            successes += 1
            match = "✓" if actual == expected else "✗"
            print(f"    {match} actual: {actual} | path: {path} | latency: {latency}ms")

        results.append(result)

        # Save incremental backup every 10 queries
        if i % 10 == 0:
            save_backup(results, BACKUP_CSV_PATH)

        # Delay — skip after last query
        if i < total:
            time.sleep(DELAY_SECONDS)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"Completed at: {datetime.now().isoformat()}")
    print(f"Total queries: {total}")
    print(f"  Successes:   {successes}")
    print(f"  Errors:      {errors}")

    save_backup(results, BACKUP_CSV_PATH)

    print("\nNext steps:")
    print("  1. Wait ~30s for Airtable logs to fully propagate")
    print("  2. python evaluate_routing.py")
    print("  3. python evaluate_quality.py")
    print("  4. python calculate_cost_savings.py")
if __name__ == "__main__":
    main()
