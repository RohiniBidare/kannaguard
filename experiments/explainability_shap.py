import os
import sys
import json
import requests
from typing import List, Dict, Tuple

API_URL = "http://127.0.0.1:8000/predict/ensemble"

SAMPLE_COMMENTS = [
    "Ellarannu nenige hakabeku",
    "Chandn ಎಕಪ್ಪ ಈಗ ಉಳ ಬಿಟ್ಟು ಕೊಂಡಿದೀಯ ನೀ ಎಸ್ಪ ದುಡ್ಡು ತಗೊಂಡಿಯ",
    "Taj Mahal story henge agidu.. hindu palace ge bandu akrama madkondu mele gommata madi adhu muslim.",
    "ಹಿಂದೂ ದೇವಾಲಯಗಳನ್ನು ಕೆಡವಿ ಕಟ್ಟಿದಾರೆ",
    "En guru idu thumba worst aagide",
    "Super video sir, nice explanation"
]

def query_endpoint(text: str) -> Dict:
    try:
        res = requests.post(API_URL, json={"text": text}, timeout=10)
        if res.status_code == 200:
            return res.json()
        return {}
    except Exception as e:
        print(f"API Error for text '{text}': {e}")
        return {}

def extract_target_score(res: Dict) -> Tuple[float, str]:
    if not res or not isinstance(res, dict):
        return 0.0, "Unknown"

    tier = res.get("tier", "No Action")

    # Safe lookup for binary model
    binary = res.get("binary") or {}
    binary_scores = binary.get("scores") or {}
    if "Toxic" in binary_scores:
        return float(binary_scores["Toxic"]), tier

    # Safe lookup for threeclass model
    threeclass = res.get("threeclass") or {}
    tc_scores = threeclass.get("scores") or {}
    if "Neutral" in tc_scores:
        return float(1.0 - tc_scores["Neutral"]), tier

    confidence = binary.get("confidence") or threeclass.get("confidence") or 0.0
    return float(confidence), tier

def compute_perturbation_attribution(text: str) -> Dict:
    base_res = query_endpoint(text)
    base_score, base_tier = extract_target_score(base_res)
    words = text.strip().split()
    attributions = []

    if len(words) <= 1:
        return {
            "text": text,
            "base_tier": base_tier,
            "base_score": round(base_score, 4),
            "tokens": [{"token": text, "importance_delta": round(base_score, 4), "relative_attribution_pct": 100.0}]
        }

    for idx, target_word in enumerate(words):
        perturbed_tokens = [w for i, w in enumerate(words) if i != idx]
        perturbed_text = " ".join(perturbed_tokens)
        
        perturbed_res = query_endpoint(perturbed_text)
        perturbed_score, _ = extract_target_score(perturbed_res)
        delta = base_score - perturbed_score

        attributions.append({
            "token": target_word,
            "importance_delta": round(delta, 4),
            "occluded_score": round(perturbed_score, 4),
            "relative_attribution_pct": 0.0
        })

    total_pos = sum(max(0.0, a["importance_delta"]) for a in attributions)
    for a in attributions:
        if total_pos > 0 and a["importance_delta"] > 0:
            a["relative_attribution_pct"] = round((a["importance_delta"] / total_pos) * 100, 2)

    return {
        "text": text,
        "base_tier": base_tier,
        "base_score": round(base_score, 4),
        "tokens": attributions
    }

def main():
    all_attributions = []
    for sentence in SAMPLE_COMMENTS:
        exp = compute_perturbation_attribution(sentence)
        all_attributions.append(exp)
        print(f"[✓] Evaluated: {sentence[:35]}... -> Tier: {exp['base_tier']}, Score: {exp['base_score']}")

    output_path = os.path.join(os.path.dirname(__file__), "explainability_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_attributions, f, ensure_ascii=False, indent=2)
    print(f"\nSuccessfully wrote verified attributions to: {output_path}")

if __name__ == "__main__":
    main()