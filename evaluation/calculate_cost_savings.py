"""
calculate_cost_savings.py
--------------------------
Answers: How much did the router save vs sending everything to a premium model?

Pulls token counts from Airtable. Calculates actual cost with the router
vs a baseline cost of sending every query to GPT-4o.

PRICING USED (verify against provider pages before publishing):
  Path A — Gemini 2.5 Flash (Google AI Studio, as of mid-2025):
    $0.15 / 1M input tokens | $0.60 / 1M output tokens

  Path B/C — Llama 3.3 70B (self-hosted proxy rate via Together AI):
    $0.88 / 1M input tokens | $0.88 / 1M output tokens
    NOTE: Groq free tier is $0 in this project. These figures represent
    what you would pay to self-host or use a paid inference provider —
    a realistic production deployment cost.

  Baseline — GPT-4o (universally recognized premium model):
    $2.50 / 1M input tokens | $10.00 / 1M output tokens

Usage:
    python calculate_cost_savings.py
"""

import os
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
AIRTABLE_PAT      = os.environ["AIRTABLE_PAT"]
AIRTABLE_BASE_ID  = os.environ["AIRTABLE_BASE_ID"]
AIRTABLE_TABLE_ID = os.environ["AIRTABLE_TABLE_ID"]

# Cost per 1 million tokens (USD)
PRICES = {
    "gemini-2.5-flash": {
        "input":  0.15,
        "output": 0.60,
        "label":  "Gemini 2.5 Flash (Google AI Studio)",
    },
    "gemini-2.0-flash": {
        "input":  0.10,
        "output": 0.40,
        "label":  "Gemini 2.0 Flash (Google AI Studio)",
    },
    "llama-3.3-70b-versatile": {
        "input":  0.88,
        "output": 0.88,
        "label":  "Llama 3.3 70B (self-hosted proxy, Together AI rate)",
    },
}

BASELINE = {
    "model":  "GPT-4o",
    "input":  2.50,
    "output": 10.00,
}

# Routing path → model fallback (used if model_used column is missing in Airtable)
PATH_TO_MODEL = {
    "A": "gemini-2.5-flash",
    "B": "llama-3.3-70b-versatile",
    "C": "llama-3.3-70b-versatile",
}


# ── Airtable Fetch ────────────────────────────────────────────────────────────
def fetch_airtable_records() -> pd.DataFrame:
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
    rows = [r["fields"] for r in records]
    return pd.DataFrame(rows)


# ── Cost Calculation ──────────────────────────────────────────────────────────
def resolve_model(row: pd.Series) -> str:
    """
    Resolve the model used for this row.
    Uses model_used column if present, otherwise derives from routing_path.
    """
    model = row.get("model_used") or row.get("agent_used") or ""
    if not model or str(model).strip() == "":
        path  = str(row.get("routing_path", "C")).upper()
        model = PATH_TO_MODEL.get(path, "llama-3.3-70b-versatile")
    return str(model).strip().lower()


def token_cost(tokens_in: float, tokens_out: float, price: dict) -> float:
    return (tokens_in / 1_000_000) * price["input"] + \
           (tokens_out / 1_000_000) * price["output"]


def calculate(df: pd.DataFrame) -> None:
    # Require token counts and routing path
    required = ["token_count_input", "token_count_output", "routing_path"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"✗ Missing columns: {missing}. Check your n8n logging schema.")
        sys.exit(1)

    # Filter rows with valid token data
    df = df[
        df["token_count_input"].notna() &
        df["token_count_output"].notna() &
        (df["token_count_input"] > 0)
    ].copy()

    if len(df) == 0:
        print("No rows with token count data found. "
              "Ensure token_count_input / token_count_output are logged.")
        sys.exit(1)

    print(f"\nAnalysing {len(df)} rows with valid token data.")

    actual_cost   = 0.0
    baseline_cost = 0.0
    unknown_models = set()

    for _, row in df.iterrows():
        tin   = float(row["token_count_input"])
        tout  = float(row["token_count_output"])
        model = resolve_model(row)

        if model in PRICES:
            actual_cost += token_cost(tin, tout, PRICES[model])
        else:
            unknown_models.add(model)
            # Fall back to cheapest known price to avoid over-stating savings
            actual_cost += token_cost(tin, tout, PRICES["llama-3.3-70b-versatile"])

        baseline_cost += token_cost(tin, tout, BASELINE)

    if unknown_models:
        print(f"\n⚠ Unknown models (used cheapest fallback): {unknown_models}")
        print("  Update PRICES dict in this script to include them.")

    saved     = baseline_cost - actual_cost
    pct_saved = (saved / baseline_cost * 100) if baseline_cost > 0 else 0

    # ── Print Results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COST ANALYSIS")
    print("=" * 60)
    print(f"Queries analysed:             {len(df)}")
    print(f"Baseline model:               {BASELINE['model']}")
    print(f"  ${BASELINE['input']:.2f}/M input | ${BASELINE['output']:.2f}/M output")
    print()
    print(f"Actual cost  (with router):   ${actual_cost:.4f}")
    print(f"Baseline cost (all {BASELINE['model']}): ${baseline_cost:.4f}")
    print(f"Cost saved:                   ${saved:.4f} ({pct_saved:.1f}%)")

    # Projected savings at scale
    if len(df) > 0:
        per_query_saving = saved / len(df)
        monthly_10k      = per_query_saving * 10_000 * 30
        print(f"\nProjected savings at 10,000 queries/day:")
        print(f"  Monthly:  ${monthly_10k:,.2f}")
        print(f"  Annually: ${monthly_10k * 12:,.2f}")

    # ── Per-Path Breakdown ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PER-PATH BREAKDOWN")
    print("=" * 60)

    path_labels = {
        "A": "Support  → Llama 3.1 8B",
        "B": "Technical → Gemini 2.5 Flash",
        "C": "Fallback  → Llama 3.3 70B",
    }
    header = f"{'Path':<32} {'n':>5} {'% queries':>10} {'Actual $':>10} {'Baseline $':>11} {'Saved $':>9}"
    print(header)
    print("-" * len(header))

    for path_id, label in path_labels.items():
        subset = df[df["routing_path"] == path_id]
        if len(subset) == 0:
            print(f"{label:<32} {'0':>5} {'0.0%':>10} {'$0.0000':>10} {'$0.0000':>11} {'$0.0000':>9}")
            continue

        pct     = len(subset) / len(df) * 100
        p_act   = sum(
            token_cost(
                float(r["token_count_input"]),
                float(r["token_count_output"]),
                PRICES.get(resolve_model(r), PRICES["llama-3.3-70b-versatile"])
            )
            for _, r in subset.iterrows()
        )
        p_base  = sum(
            token_cost(float(r["token_count_input"]), float(r["token_count_output"]), BASELINE)
            for _, r in subset.iterrows()
        )
        p_saved = p_base - p_act

        print(f"{label:<32} {len(subset):>5} {pct:>9.1f}% "
              f"${p_act:>9.4f} ${p_base:>10.4f} ${p_saved:>8.4f}")

    # ── Token Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)
    total_in  = df["token_count_input"].sum()
    total_out = df["token_count_output"].sum()
    avg_in    = df["token_count_input"].mean()
    avg_out   = df["token_count_output"].mean()
    print(f"Total input tokens:  {total_in:,.0f} (avg {avg_in:.0f}/query)")
    print(f"Total output tokens: {total_out:,.0f} (avg {avg_out:.0f}/query)")

    print("\n" + "=" * 60)
    print("Cost figures are for README Section 8 — Business Impact.")
    print("Verify pricing at provider pages before publishing.")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("COST SAVINGS ANALYSIS — calculate_cost_savings.py")
    print("=" * 60)

    df = fetch_airtable_records()
    calculate(df)


if __name__ == "__main__":
    main()