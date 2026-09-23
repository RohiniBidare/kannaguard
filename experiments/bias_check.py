"""
Formal Bias Check using Search Topic metadata

Uses the "Search Topic" column already present in your original dataset
to quantify whether certain topics (e.g. religious/communal) get
disproportionately flagged by the model compared to others (e.g.
travel, food, entertainment) - a formal, data-driven version of the
topic-bias finding discovered during live testing.

Requires: your labeled dataset CSV with columns 'Comment' and 'Search Topic'
"""

import pandas as pd
import requests
import time

API_URL = "http://127.0.0.1:8000/predict/ensemble"
from pathlib import Path

DATASET_FILE = Path(__file__).resolve().parent / "final_dataset_cohen_kappa_above_93.csv"
MAX_PER_TOPIC = 50   # cap per topic to keep runtime reasonable


def classify(text):
    try:
        r = requests.post(API_URL, json={"text": str(text)})
        result = r.json()
        return result.get("tier", "ERROR")
    except Exception:
        return "ERROR"


def main():
    df = pd.read_csv(DATASET_FILE)

    if "Search Topic" not in df.columns:
        print("ERROR: 'Search Topic' column not found in this file.")
        print("Available columns:", df.columns.tolist())
        return

    topics = df["Search Topic"].dropna().unique()
    print(f"Found {len(topics)} distinct search topics.\n")

    topic_results = []

    for topic in topics:
        topic_df = df[df["Search Topic"] == topic].sample(
            n=min(MAX_PER_TOPIC, len(df[df["Search Topic"] == topic])),
            random_state=42
        )

        tiers = {"Auto-Flag": 0, "Needs Review": 0, "No Action": 0, "ERROR": 0}

        for comment in topic_df["Comment"]:
            tier = classify(comment)
            tiers[tier] = tiers.get(tier, 0) + 1
            time.sleep(0.05)  # light throttle

        total = sum(tiers.values())
        flagged_rate = round((tiers["Auto-Flag"] + tiers["Needs Review"]) / total * 100, 1) if total else 0

        print(f"[{topic}] n={total} | Auto-Flag={tiers['Auto-Flag']} | "
              f"Needs Review={tiers['Needs Review']} | No Action={tiers['No Action']} | "
              f"Flagged rate={flagged_rate}%")

        topic_results.append({
            "topic": topic,
            "n": total,
            "auto_flag": tiers["Auto-Flag"],
            "needs_review": tiers["Needs Review"],
            "no_action": tiers["No Action"],
            "flagged_rate_pct": flagged_rate,
        })

    results_df = pd.DataFrame(topic_results).sort_values("flagged_rate_pct", ascending=False)

    print("\n" + "=" * 70)
    print("TOPICS RANKED BY FLAGGED RATE (highest first)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_csv("bias_check_results.csv", index=False)
    print("\nSaved to bias_check_results.csv")

    # Simple spread statistic - shows how much flagged rate varies by topic
    spread = results_df["flagged_rate_pct"].max() - results_df["flagged_rate_pct"].min()
    print(f"\nFlagged-rate spread across topics: {spread:.1f} percentage points")
    print("A large spread indicates the model's flagging behavior is topic-dependent,")
    print("consistent with the topic-bias pattern observed during live testing.")


if __name__ == "__main__":
    main()