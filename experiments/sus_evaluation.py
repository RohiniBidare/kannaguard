import os
import pandas as pd
import numpy as np

# Mock/Trial evaluation cohort: 5 evaluators (2 bilingual community moderators, 3 digital content creators)
EVAL_DATA = [
    {"user": "Moderator_1 (News)",     "q1": 5, "q2": 1, "q3": 5, "q4": 1, "q5": 4, "q6": 2, "q7": 5, "q8": 1, "q9": 4, "q10": 1},
    {"user": "Moderator_2 (Debate)",   "q1": 4, "q2": 2, "q3": 4, "q4": 1, "q5": 5, "q6": 2, "q7": 4, "q8": 2, "q9": 4, "q10": 1},
    {"user": "Creator_1 (Tech/Vlog)",  "q1": 5, "q2": 1, "q3": 4, "q4": 1, "q5": 4, "q6": 1, "q7": 5, "q8": 1, "q9": 5, "q10": 2},
    {"user": "Creator_2 (Entertainment)", "q1": 4, "q2": 2, "q3": 4, "q4": 2, "q5": 4, "q6": 2, "q7": 4, "q8": 1, "q9": 4, "q10": 1},
    {"user": "Creator_3 (Education)",  "q1": 5, "q2": 1, "q3": 5, "q4": 1, "q5": 5, "q6": 2, "q7": 5, "q8": 1, "q9": 4, "q10": 1}
]

def calculate_sus(responses):
    records = []
    odd_keys = ["q1", "q3", "q5", "q7", "q9"]
    even_keys = ["q2", "q4", "q6", "q8", "q10"]

    for r in responses:
        # Odd items: score - 1
        pos_contrib = sum(r[k] - 1 for k in odd_keys)
        # Even items: 5 - score
        neg_contrib = sum(5 - r[k] for k in even_keys)
        
        raw_sum = pos_contrib + neg_contrib
        sus_score = raw_sum * 2.5
        records.append({**r, "raw_sum": raw_sum, "sus_score": sus_score})

    df = pd.DataFrame(records)
    return df

def main():
    print("=" * 65)
    print("KANNAGUARD: SYSTEM USABILITY SCALE (SUS) EVALUATION")
    print("=" * 65)

    df = calculate_sus(EVAL_DATA)
    mean_sus = df["sus_score"].mean()
    std_sus = df["sus_score"].std()

    print(df[["user", "raw_sum", "sus_score"]].to_string(index=False))
    print("-" * 65)
    print(f"Mean SUS Score:       {mean_sus:.2f} (Target > 68.0)")
    print(f"Standard Deviation:   {std_sus:.2f}")
    
    if mean_sus >= 80.3:
        grade = "A (Excellent Usability)"
    elif mean_sus >= 68.0:
        grade = "B (Above Average Usability)"
    else:
        grade = "Marginal / Needs Improvement"
    print(f"Qualitative Rating:   {grade}")

    out_csv = os.path.join(os.path.dirname(__file__), "sus_survey_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n[✓] Results exported to: {out_csv}")

if __name__ == "__main__":
    main()