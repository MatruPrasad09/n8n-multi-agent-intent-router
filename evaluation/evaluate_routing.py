"""
evaluate_routing.py
-------------------
Answers: Did the router send queries to the correct path?

Deterministic — zero LLM calls. Run this first, immediately after the
golden set batch completes. Pulls data directly from Airtable.

Outputs:
  - 3-class confusion matrix (support / technical / unknown)
  - Classification report (precision, recall, F1 per class)
  - Confidence threshold analysis
  - Adversarial query handling stats
  - Latency analysis (P50 / P95 per path)

Usage:
    python evaluate_routing.py
"""

import os
import sys

import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
AIRTABLE_PAT      = os.environ["AIRTABLE_PAT"]
AIRTABLE_BASE_ID  = os.environ["AIRTABLE_BASE_ID"]
AIRTABLE_TABLE_ID = os.environ["AIRTABLE_TABLE_ID"]

LABELS = ["support", "technical", "unknown"]

# ── Airtable Fetch ────────────────────────────────────────────────────────────
def fetch_airtable_records() -> pd.DataFrame:
    """Fetch all records from Airtable with automatic pagination."""
    url     = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_TABLE_ID}"
    headers = {"Authorization": f"Bearer {AIRTABLE_PAT}"}
    records = []
    params  = {"pageSize": 100}

    print("Fetching records from Airtable...", end="", flush=True)
    while True:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        records.extend(data.get("records", []))
        print(".", end="", flush=True)
        offset = data.get("offset")
        if not offset:
            break
        params["offset"] = offset

    print(f" {len(records)} records fetched.")

    if not records:
        print("No records found in Airtable. Have you run run_golden_set.py?")
        sys.exit(1)

    # Flatten the Airtable record structure: {"id":..., "fields":{...}}
    rows = [r["fields"] for r in records]
    return pd.DataFrame(rows)


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "request_id", "expected_intent", "actual_intent",
        "confidence_score", "routing_path", "is_fallback_route",
        "router_latency_ms", "agent_latency_ms", "total_latency_ms",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n✗ Missing columns in Airtable: {missing}")
        print("  Check your n8n workflow logging schema and re-run the golden set.")
        sys.exit(1)

    # Only evaluate rows where ground truth is known (golden set rows)
    df_eval = df[df["expected_intent"].notna()].copy()
    print(f"Rows with ground truth (golden set): {len(df_eval)} / {len(df)} total")

    if len(df_eval) == 0:
        print("No rows with expected_intent found. Did you pass expected_intent in the batch?")
        sys.exit(1)

    return df_eval


# ── Analysis Functions ────────────────────────────────────────────────────────
def print_confusion_matrix(df: pd.DataFrame) -> None:
    y_true = df["expected_intent"]
    y_pred = df["actual_intent"]
    cm     = confusion_matrix(y_true, y_pred, labels=LABELS)

    print("\n" + "=" * 60)
    print("ROUTING CONFUSION MATRIX (3-class)")
    print("=" * 60)
    print(f"{'':20}", end="")
    for label in LABELS:
        print(f"pred_{label:10}", end="")
    print()
    for i, label in enumerate(LABELS):
        print(f"true_{label:15}", end="")
        for j in range(len(LABELS)):
            print(f"{cm[i][j]:>15}", end="")
        print()

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    # Overall accuracy
    correct = (y_true == y_pred).sum()
    total   = len(df)
    print(f"Overall routing accuracy: {correct}/{total} ({100*correct/total:.1f}%)")


def print_confidence_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 60)

    fallback_count = df["is_fallback_route"].sum()
    print(f"Fallback route (Path C): {fallback_count}/{len(df)} ({100*fallback_count/len(df):.1f}%)")

    df["is_correct"] = df["expected_intent"] == df["actual_intent"]
    correct_conf   = df[df["is_correct"]]["confidence_score"].mean()
    incorrect_conf = df[~df["is_correct"]]["confidence_score"].mean()
    print(f"Mean confidence — correct routes:   {correct_conf:.3f}")
    print(f"Mean confidence — incorrect routes: {incorrect_conf:.3f}")

    # Threshold sensitivity
    print("\nThreshold sensitivity (queries reaching an agent vs falling back):")
    for threshold in [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        above = (df["confidence_score"] >= threshold).sum()
        print(f"  threshold={threshold:.2f}: {above}/{len(df)} queries routed to agent "
              f"({100*above/len(df):.0f}%)")


def print_adversarial_analysis(df: pd.DataFrame) -> None:
    adv = df[df["expected_intent"] == "unknown"]
    if len(adv) == 0:
        return
    print("\n" + "=" * 60)
    print(f"ADVERSARIAL QUERY ANALYSIS ({len(adv)} queries)")
    print("=" * 60)

    correctly_suppressed = (adv["actual_intent"] == "unknown").sum()
    leaked_to_agent      = (adv["actual_intent"] != "unknown").sum()
    print(f"Correctly returned 'unknown' (suppressed): {correctly_suppressed}/{len(adv)}")
    print(f"Incorrectly routed to an agent (leaked):   {leaked_to_agent}/{len(adv)}")

    if leaked_to_agent > 0:
        leaked = adv[adv["actual_intent"] != "unknown"]
        print("\nLeaked adversarial queries:")
        for _, row in leaked.iterrows():
            print(f"  [{row.get('request_id','?')}] '{row['user_query'][:60]}...'")
            print(f"    → routed to: {row['actual_intent']} (confidence: {row['confidence_score']:.2f})")


def print_latency_analysis(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("LATENCY ANALYSIS (milliseconds)")
    print("=" * 60)
    path_labels = {
        "A": "Support — Gemini 2.5 Flash",
        "B": "Technical — Groq Llama 70B",
        "C": "Fallback — Groq Llama 70B",
    }
    header = f"{'Path':<35} {'n':>5} {'P50':>8} {'P95':>8} {'Max':>8}"
    print(header)
    print("-" * len(header))

    for path_id, label in path_labels.items():
        subset = df[df["routing_path"] == path_id]["total_latency_ms"].dropna()
        if len(subset) == 0:
            print(f"{label:<35} {'0':>5} {'—':>8} {'—':>8} {'—':>8}")
            continue
        print(
            f"{label:<35} {len(subset):>5} "
            f"{subset.quantile(0.50):>7.0f}ms "
            f"{subset.quantile(0.95):>7.0f}ms "
            f"{subset.max():>7.0f}ms"
        )

    # Router latency separately
    router_lat = df["router_latency_ms"].dropna()
    if len(router_lat):
        print(f"\nRouter only (Groq 8B classification):")
        print(f"  P50: {router_lat.quantile(0.50):.0f}ms | "
              f"P95: {router_lat.quantile(0.95):.0f}ms | "
              f"n={len(router_lat)}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("ROUTING EVALUATION — evaluate_routing.py")
    print("=" * 60)

    df      = fetch_airtable_records()
    df_eval = validate_columns(df)

    print_confusion_matrix(df_eval)
    print_confidence_analysis(df_eval)
    print_adversarial_analysis(df_eval)
    print_latency_analysis(df_eval)

    print("\n" + "=" * 60)
    print("Next: python evaluate_quality.py")
    print("=" * 60)


if __name__ == "__main__":
    main()