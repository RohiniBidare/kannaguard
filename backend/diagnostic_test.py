import time
import requests
import statistics

API_URL = "http://127.0.0.1:8000/predict/ensemble"

TEST_COMMENTS = [
    "Ellarannu nenige hakabeku",
    "Namaskara, channagideera?",
    "Super video sir, nice explanation",
    "Naanu ee channel subscribe madidini",
    "Thank you for the information",
    "Chandn ಎಕಪ್ಪ ಈಗ ಉಳ ಬಿಟ್ಟು ಕೊಂಡಿದೀಯ ನೀ ಎಸ್ಪ ದುಡ್ಡು ತಗೊಂಡಿಯ",
    "En guru idu thumba worst aagide",
    "Namma Karnataka namma hecchume"
]

def check_backend_alive():
    try:
        res = requests.post(API_URL, json={"text": "test ping"}, timeout=5)
        return res.status_code == 200
    except requests.exceptions.RequestException:
        return False

def run_single_inference_benchmark(iterations=30):
    print(f"\n[1/2] Running single-inference latency test ({iterations} rounds)...")
    latencies = []
    
    # Warmup run (ignores initial cold start)
    requests.post(API_URL, json={"text": "warmup"}, timeout=10)
    
    for i in range(iterations):
        comment = TEST_COMMENTS[i % len(TEST_COMMENTS)]
        start = time.perf_counter()
        res = requests.post(API_URL, json={"text": comment}, timeout=10)
        end = time.perf_counter()
        
        if res.status_code == 200:
            latencies.append((end - start) * 1000)  # milliseconds
        else:
            print(f"Warning: Request failed with status {res.status_code}")
            
    return latencies

def run_batch_simulation(batch_sizes=[10, 50, 100]):
    print("\n[2/2] Running simulated batch comment scans...")
    results = {}
    
    for size in batch_sizes:
        print(f"  -> Testing batch size of {size} sequential comments...")
        start = time.perf_counter()
        success = 0
        for i in range(size):
            comment = TEST_COMMENTS[i % len(TEST_COMMENTS)]
            res = requests.post(API_URL, json={"text": comment}, timeout=10)
            if res.status_code == 200:
                success += 1
        elapsed = time.perf_counter() - start
        
        cpm = (success / elapsed) * 60 if elapsed > 0 else 0
        results[size] = {
            "elapsed_sec": elapsed,
            "success": success,
            "cpm": cpm,
            "avg_per_comment_sec": elapsed / success if success > 0 else 0
        }
    return results

def main():
    print("=" * 65)
    print("KANNAGUARD: INFERENCE SPEED & THROUGHPUT BENCHMARK")
    print("=" * 65)
    
    if not check_backend_alive():
        print("ERROR: Backend is not responding at http://127.0.0.1:8000")
        print("Please ensure your FastAPI / Uvicorn server is actively running.")
        return

    latencies = run_single_inference_benchmark(iterations=30)
    
    if not latencies:
        print("Benchmark failed: No successful responses received.")
        return

    mean_lat = statistics.mean(latencies)
    median_lat = statistics.median(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    print("\n--- Latency Breakdown (Per Comment) ---")
    print(f"Mean Latency:    {mean_lat:.2f} ms")
    print(f"Median Latency:  {median_lat:.2f} ms")
    print(f"Min Latency:     {min_lat:.2f} ms")
    print(f"Max Latency:     {max_lat:.2f} ms")

    batch_results = run_batch_simulation(batch_sizes=[10, 50, 100])
    
    print("\n--- Batch Throughput Breakdown ---")
    print(f"{'Batch Size':<12} | {'Time (s)':<10} | {'Comments/Sec':<14} | {'Comments/Min':<14}")
    print("-" * 58)
    for size, data in batch_results.items():
        cps = data['success'] / data['elapsed_sec']
        print(f"{size:<12} | {data['elapsed_sec']:<10.2f} | {cps:<14.2f} | {data['cpm']:<14.2f}")

    print("\n" + "=" * 65)
    print("Paper Writeup Reference:")
    print(f"Mean inference latency: {mean_lat:.2f} ms per comment.")
    print(f"100-comment scan completion time: {batch_results[100]['elapsed_sec']:.2f} seconds.")
    print("=" * 65)

if __name__ == "__main__":
    main()