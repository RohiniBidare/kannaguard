"""
Tests the ensemble endpoint against the exact false-positive examples
found earlier, to confirm the mitigation layer is working.
"""
import requests

API_BASE = "http://127.0.0.1:8000"

test_comments = [
    "ನಮ್ಮ ವಿಜಯಪುರ❤️",       # false positive earlier - should now be No Action or Needs Review
    "❤❤❤",                   # emoji only - should be No Action (pre-filtered)
    "Chatrapati Shivaji Maharaj ki Jai",  # false positive earlier
    "ಸೂಳೆ ಮಗ ನೀನು",          # genuinely hostile - should still Auto-Flag
]

for c in test_comments:
    r = requests.post(f"{API_BASE}/predict/ensemble", json={"text": c}).json()
    print(f"Comment: {c}")
    print(f"  Tier: {r['tier']} | Final label: {r['final_label']}")
    print(f"  Action: {r['recommended_action']}")
    print()
