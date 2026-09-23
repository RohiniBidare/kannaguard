import time
import requests
import concurrent.futures
import statistics

API_URL = "http://127.0.0.1:8000/predict/ensemble"

TEST_COMMENTS = [
    "Ellarannu nenige hakabeku",
    "Namaskara, channagideera?",
    "Super video sir, nice explanation",
    "Chandn ಎಕಪ್ಪ ಈಗ ಉಳ ಬಿಟ್ಟು ಕೊಂಡಿದೀಯ ನೀ ಎಸ್ಪ ದುಡ್ಡು ತಗೊಂಡಿಯ",
    "En guru idu thumba worst aagide"
]

def send_request(idx):
    comment = TEST_COMMENTS[idx % len(TEST_COMMENTS)]
    start = time.perf_counter()
    try:
        res = requests.post(API_URL, json={"text": comment}, timeout=15)
        latency = (time.perf_counter() - start) * 1000
        return {"status": res.status_code, "latency": latency, "success": res.status_code == 200}
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return {"status": "error", "latency": latency, "success": False, "error": str(e)}

def run_concurrent_load(concurrency_level, total_requests):
    print(f"\nTesting {total_requests} requests across {concurrency_level} concurrent workers...")
    
    start_total = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(send_request, i) for i in range(total_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    total_time = time.perf_counter() - start_total

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in successes]

    success_rate = (len(successes) / total_requests) * 100
    throughput = len(successes) / total_time if total_time > 0 else 0
    avg_lat = statistics.mean(latencies) if latencies else 0
    p95_lat = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else (max(latencies) if latencies else 0)

    print(f"  Success Rate: {success_rate:.1f}% ({len(successes)}/{total_requests})")
    print(f"  Throughput:   {throughput:.2f} requests/sec")
    print(f"  Avg Latency:  {avg_lat:.2f} ms")
    print(f"  p95 Latency:  {p95_lat:.2f} ms")
    
    return {
        "concurrency": concurrency_level,
        "success_rate": success_rate,
        "throughput": throughput,
        "avg_lat": avg_lat,
        "p95_lat": p95_lat
    }

def main():
    print("=" * 65)
    print("KANNAGUARD: CONCURRENT LOAD & STRESS TEST")
    print("=" * 65)

    test_runs = [
        {"concurrency": 2, "requests": 20},
        {"concurrency": 5, "requests": 30},
        {"concurrency": 10, "requests": 40}
    ]

    for run in test_runs:
        run_concurrent_load(run["concurrency"], run["requests"])

    print("\n" + "=" * 65)
    print("Benchmark complete. Include these throughput figures in your systems section.")
    print("=" * 65)

if __name__ == "__main__":
    main()