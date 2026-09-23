import json
import os
import matplotlib.pyplot as plt

INPUT_FILE = os.path.join(os.path.dirname(__file__), "explainability_results.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

cases = [
    ("En guru idu thumba worst aagide", "xai_insult_isolation.png"),
    ("Taj Mahal story henge agidu.. hindu palace ge bandu akrama madkondu mele gommata madi adhu muslim.", "xai_demographic_bias.png")
]

for target_text, filename in cases:
    record = next((item for item in data if item["text"].strip() == target_text.strip()), None)
    if not record:
        print(f"[!] Warning: Could not find exact text match for: {target_text}")
        continue

    # Take top 8 tokens by absolute delta impact
    tokens_data = sorted(record["tokens"], key=lambda x: abs(x["importance_delta"]), reverse=True)[:8]
    tokens_data.sort(key=lambda x: x["importance_delta"])  # sort ascending for horizontal bars

    tokens = [t["token"] for t in tokens_data]
    impacts = [t["importance_delta"] for t in tokens_data]
    colors = ["#d9534f" if imp > 0 else "#5bc0de" for imp in impacts]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(tokens, impacts, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xlabel("Marginal Impact on Toxic Score (Δ)", fontsize=11, fontweight="bold")
    ax.set_title(f"Token Attribution: \"{record['text'][:40]}...\"\nTier: {record['base_tier']} | Base P(toxic) = {record['base_score']:.4f}",
                 fontsize=11, pad=12)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    # Dynamic limits so small deltas still render prominently
    max_val = max(abs(min(impacts, default=0)), abs(max(impacts, default=0)))
    bound = max(max_val * 1.3, 0.05)
    ax.set_xlim(-bound, bound)

    for bar in bars:
        width = bar.get_width()
        ha = "left" if width >= 0 else "right"
        offset = bound * 0.02 if width >= 0 else -bound * 0.02
        ax.annotate(f"{width:+.4f}",
                    xy=(width + offset, bar.get_y() + bar.get_height() / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha=ha, va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Successfully generated non-empty chart: {save_path}")