import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Base paths to your local models
BINARY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "models", "indicbertv2_binary_final")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "quantized_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BENCHMARK_SENTENCES = [
    "En guru idu thumba worst aagide",
    "Super video sir, nice explanation",
    "Taj Mahal story henge agidu.. hindu palace ge bandu",
    "Chandn ಎಕಪ್ಪ ಈಗ ಉಳ ಬಿಟ್ಟು ಕೊಂಡಿದೀಯ",
    "Nanu cinema nodoke hogthini ivattu",
    "Tumba olle content bro, continue maadi"
]

def get_dir_size_mb(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)

def benchmark_inference(model, tokenizer, sentences, runs=20):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # Warmup
    inputs = tokenizer(sentences[0], return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        for _ in range(5):
            _ = model(**inputs)

    latencies = []
    for _ in range(runs):
        for s in sentences:
            inputs = tokenizer(s, return_tensors="pt", truncation=True, max_length=128)
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            latencies.append((time.perf_counter() - start) * 1000)

    return sum(latencies) / len(latencies)

def main():
    print("=" * 65)
    print("KANNAGUARD: MODEL DISTILLATION & DYNAMIC INT8 QUANTIZATION")
    print("=" * 65)

    if not os.path.exists(BINARY_MODEL_PATH):
        print(f"[!] Path not found: {BINARY_MODEL_PATH}")
        return

    print("[*] Loading FP32 Baseline PyTorch Model...")
    tokenizer = AutoTokenizer.from_pretrained(BINARY_MODEL_PATH)
    fp32_model = AutoModelForSequenceClassification.from_pretrained(BINARY_MODEL_PATH)

    # 1. Benchmark FP32
    print("[*] Benchmarking Baseline FP32 Model on CPU...")
    fp32_latency = benchmark_inference(fp32_model, tokenizer, BENCHMARK_SENTENCES)
    fp32_size = get_dir_size_mb(BINARY_MODEL_PATH)

    # 2. Dynamic PyTorch INT8 Quantization
    print("[*] Applying Dynamic INT8 Quantization to Linear Layers...")
    int8_model = torch.quantization.quantize_dynamic(
        fp32_model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Save Quantized Weights
    quant_save_path = os.path.join(OUTPUT_DIR, "indicbertv2_binary_int8.pt")
    torch.save(int8_model.state_dict(), quant_save_path)
    int8_size = os.path.getsize(quant_save_path) / (1024 * 1024)

    # 3. Benchmark INT8
    print("[*] Benchmarking Quantized INT8 Model on CPU...")
    int8_latency = benchmark_inference(int8_model, tokenizer, BENCHMARK_SENTENCES)

    # Computations
    compression_pct = ((fp32_size - int8_size) / fp32_size) * 100
    speedup = fp32_latency / int8_latency

    print("\n" + "=" * 65)
    print("OPTIMIZATION & COMPRESSION BENCHMARK RESULTS")
    print("=" * 65)
    print(f"Original Model Size (FP32):   {fp32_size:.2f} MB")
    print(f"Quantized Model Size (INT8):  {int8_size:.2f} MB")
    print(f"Storage Reduction:            {compression_pct:.1f}%")
    print("-" * 65)
    print(f"FP32 Mean CPU Latency:        {fp32_latency:.2f} ms")
    print(f"INT8 Mean CPU Latency:        {int8_latency:.2f} ms")
    print(f"Inference Speedup Factor:     {speedup:.2f}x")
    print("=" * 65)

if __name__ == "__main__":
    main()