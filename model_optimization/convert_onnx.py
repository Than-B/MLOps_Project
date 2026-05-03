"""
Step 3: Convert PyTorch Model → ONNX Format
ใช้ torch.onnx.export เพื่อแปลงโมเดลและวัดผล
"""

import time
import json
import statistics
from pathlib import Path

import torch
import onnxruntime as ort
from PIL import Image
from transformers import AutoFeatureExtractor, AutoModelForImageClassification


# ─── Config ───────────────────────────────────────────────
MODEL_NAME   = "dima806/facial_emotions_image_detection"
MODEL_DIR    = Path("./models/original")
ONNX_PATH    = Path("./models/onnx/model.onnx")
NUM_WARMUP   = 3
NUM_RUNS     = 20
INPUT_SHAPE  = (1, 3, 224, 224)  # batch=1, RGB, 224x224


def get_model_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 ** 2)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 2)


def get_test_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(128, 100, 90))


def main():
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("     Convert PyTorch → ONNX")
    print("=" * 60)

    # โหลด PyTorch model
    print("  โหลด PyTorch model จาก local...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
    model.eval()
    id2label = model.config.id2label

    # เตรียม dummy input
    image = get_test_image()
    inputs = extractor(images=image, return_tensors="pt")
    dummy_input = inputs["pixel_values"]  # shape: (1, 3, 224, 224)

    # ─── Export to ONNX ─────────────────────────────────────
    print(f"  กำลัง export ไปที่: {ONNX_PATH}")
    t_export_start = time.perf_counter()

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(ONNX_PATH),
            export_params=True,
            opset_version=14,          # ใช้ 14 รองรับ ops ใหม่กว่า
            do_constant_folding=True,  # optimize constant ใน graph
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={
                "pixel_values": {0: "batch_size"},
                "logits":       {0: "batch_size"},
            },
        )

    t_export_end = time.perf_counter()
    print(f"  Export เสร็จใน {(t_export_end - t_export_start):.2f} วินาที")

    # ─── วัดขนาด ────────────────────────────────────────────
    onnx_size_mb = get_model_size_mb(ONNX_PATH)
    orig_size_mb = get_model_size_mb(MODEL_DIR)
    print(f"  ขนาด ONNX  : {onnx_size_mb:.2f} MB")
    print(f"  ขนาด Original: {orig_size_mb:.2f} MB")
    print(f"  ลดลง       : {((orig_size_mb - onnx_size_mb) / orig_size_mb * 100):.1f}%")

    # ─── ONNX Inference ─────────────────────────────────────
    print()
    print("  สร้าง ONNX Runtime session...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(ONNX_PATH), sess_options=sess_options,
                                   providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    ort_input = {input_name: dummy_input.numpy()}

    # Warmup
    print(f"  Warmup {NUM_WARMUP} ครั้ง...")
    for _ in range(NUM_WARMUP):
        _ = session.run(None, ort_input)

    # Measure
    print(f"  วัดค่า Latency {NUM_RUNS} ครั้ง...")
    latencies = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        ort_outputs = session.run(None, ort_input)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    mean_lat = statistics.mean(latencies)
    p50_lat  = statistics.median(latencies)
    p95_lat  = sorted(latencies)[int(0.95 * NUM_RUNS)]
    min_lat  = min(latencies)

    # ดูผล prediction
    import numpy as np
    logits = ort_outputs[0]
    predicted_class = int(np.argmax(logits, axis=-1)[0])
    predicted_label = id2label[predicted_class]

    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │         ONNX INFERENCE RESULTS          │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Model size   : {onnx_size_mb:>8.2f} MB              │")
    print(f"  │  Mean latency : {mean_lat:>8.2f} ms              │")
    print(f"  │  P50 latency  : {p50_lat:>8.2f} ms              │")
    print(f"  │  P95 latency  : {p95_lat:>8.2f} ms              │")
    print(f"  │  Min latency  : {min_lat:>8.2f} ms              │")
    print(f"  │  Prediction   : {predicted_label:<24s} │")
    print("  └─────────────────────────────────────────┘")

    # บันทึกผล
    results = {
        "model": "ONNX",
        "size_mb": round(onnx_size_mb, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "min_latency_ms": round(min_lat, 2),
    }

    results_path = Path("./models/benchmark_results.json")
    all_results = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    all_results = [r for r in all_results if r["model"] != "ONNX"]
    all_results.append(results)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  บันทึกผลไปที่: {results_path}")


if __name__ == "__main__":
    main()
