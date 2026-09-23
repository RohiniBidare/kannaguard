"""
Speed Test

Measures how long your system takes to classify comments, broken into
stages (single comment vs batch), so you can report concrete latency
numbers in your paper.
"""

import requests
import time
import statistics

API_URL = "http://127.0.0.1:8000/predict/ensemble"

SAMPLE_COMMENTS = [
    "Super video sir, nice explanation",
    "ಸೂಳೆ ಮಗ ನೀನು",
    "Thank you for the information",
    "bro adu hindugalu kattirodu adake alli hindugala kuruvugalive",
    "Nice content please continue",
    "Namaskara, channagideera?",
    "Yenu madidiri e lofer na",
    "Please visit our village next time",
    "Taj Mahal story henge agidu hindu palace muslim",
    "This was very informative thank you",
]


def time_single_request(text):
    start = time.time()
    try:
        requests.post(API_URL, json={"text": text}, timeout=15)
    except Exception as e:
        print(f"  Error: {e}")
        return None
    return time.time() - start


print("=" * 60)
print("SPEED TEST - Single Comment Latency")
print("=" * 60)

times = []
for comment in SAMPLE_COMMENTS:
    t = time_single_request(comment)
    if t is not None:
        times.append(t)
        print(f"  {t:.3f}s  |  {comment[:50]}")

if times:
    print(f"\nMean latency:   {statistics.mean(times):.3f}s")
    print(f"Median latency: {statistics.median(times):.3f}s")
    print(f"Min / Max:      {min(times):.3f}s / {max(times):.3f}s")

print("\n" + "=" * 60)
print("SPEED TEST - Simulated Batch Scan (sequential)")
print("=" * 60)

batch_sizes = [10, 50, 100]

for size in batch_sizes:
    test_batch = (SAMPLE_COMMENTS * ((size // len(SAMPLE_COMMENTS)) + 1))[:size]
    start = time.time()
    for comment in test_batch:
        try:
            requests.post(API_URL, json={"text": comment}, timeout=15)
        except Exception:
            pass
    elapsed = time.time() - start
    per_comment = elapsed / size
    print(f"\n{size} comments: {elapsed:.2f}s total | {per_comment:.3f}s per comment | "
          f"~{60/per_comment:.0f} comments/minute")

print("\n" + "=" * 60)
print("Use these numbers in your paper's 'System Performance' section,")
print("e.g. 'The system processes approximately X comments per minute")
print("on CPU, making a Y-comment video scan take approximately Z seconds.'")
print("=" * 60)