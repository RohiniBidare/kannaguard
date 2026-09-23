import os
import requests
import pandas as pd

BACKEND_SCAN_URL = "http://127.0.0.1:8000/scan_video"

# Real, verified public Kannada YouTube video IDs
TARGET_VIDEOS = [
    {"video_id": "vyMMIh0TPyo", "label": "Suvarna News Debate"},
    {"video_id": "VJGx2QHATkY", "label": "Vistara News Political Debate"},
    {"video_id": "tSHIac079_s", "label": "Kannada Media Discussion"}
]
def run_audit(target_list):
    print("=" * 65)
    print("KANNAGUARD: LIVE YOUTUBE CONTENT AUDIT (VIA /scan_video)")
    print("=" * 65)

    all_flagged_records = []
    
    for target in target_list:
        vid = target["video_id"]
        label = target.get("label", "Video")
        print(f"\n[*] Scanning YouTube Video: '{label}' (ID: {vid}) ...")
        
        try:
            res = requests.post(BACKEND_SCAN_URL, json={"video_id": vid, "max_comments": 100}, timeout=90)
            if res.status_code != 200:
                print(f"[!] Request failed with HTTP {res.status_code}")
                continue
                
            data = res.json()
            if "error" in data:
                print(f"[!] YouTube API Error for {vid}: {data['error']}")
                continue

            total = data.get("total_comments", 0)
            auto_flag = data.get("summary", {}).get("Auto-Flag", 0)
            needs_review = data.get("summary", {}).get("Needs Review", 0)
            no_action = data.get("summary", {}).get("No Action", 0)
            live_count = data.get("flagged_still_live_count", 0)
            time_saved = data.get("time_saved_minutes", 0.0)

            print(f"  -> Total Comments Fetched:  {total}")
            print(f"  -> Auto-Flagged (Severe):    {auto_flag}")
            print(f"  -> Needs Review:             {needs_review}")
            print(f"  -> Harmless / No Action:     {no_action}")
            print(f"  -> Still Live on YouTube:    {live_count}")
            print(f"  -> Estimated Time Saved:     {time_saved} mins")

            for item in data.get("results", []):
                if item.get("tier") in ["Auto-Flag", "Needs Review"]:
                    all_flagged_records.append({
                        "video_id": vid,
                        "video_label": label,
                        "comment_text": item.get("comment"),
                        "moderation_tier": item.get("tier"),
                        "final_label": item.get("final_label"),
                        "matched_terms": ", ".join(item.get("matched_terms", [])),
                        "explanation": item.get("explanation"),
                        "still_live_on_yt": True
                    })

        except Exception as e:
            print(f"[!] Failed connecting to backend for video {vid}: {e}")

    if all_flagged_records:
        df = pd.DataFrame(all_flagged_records)
        out_path = os.path.join(os.path.dirname(__file__), "still_live_audit_results.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print("\n" + "=" * 65)
        print(f"[✓] Audit complete! Exported {len(df)} unmoderated toxic comments to:")
        print(f"    {out_path}")
        print("=" * 65)
    else:
        print("\n[i] No flagged comments found or queries failed. Check your video IDs and backend.")

if __name__ == "__main__":
    run_audit(TARGET_VIDEOS)