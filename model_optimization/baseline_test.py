"""
Step 1 & 2: Model Selection + Baseline Inference Test
Model: dima806/facial_emotions_image_detection
วัดค่า Latency และขนาดโมเดลก่อน Optimization
"""

import time
import os
import statistics
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import requests
from io import BytesIO


# ─── Config ───────────────────────────────────────────────
MODEL_NAME = "dima806/facial_emotions_image_detection"
MODEL_DIR  = Path("./models/original")
NUM_WARMUP = 3    # warmup runs (ไม่นับ)
NUM_RUNS   = 20   # จำนวนรอบที่จะวัด

# ─── Helper: ดาวน์โหลดภาพทดสอบ ────────────────────────────
def get_test_image() -> Image.Image:
    """ดาวน์โหลดภาพใบหน้าตัวอย่างจาก web"""
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg"
    # ใช้ภาพ solid color แทนเพื่อไม่ง้อ internet ใน CI
    img = Image.new("RGB", (224, 224), color=(128, 100, 90))
    return img


def get_model_size_mb(path: Path) -> float:
    """คำนวณขนาดโฟลเดอร์หรือไฟล์เป็น MB"""
    if path.is_file():
        return path.stat().st_size / (1024 ** 2)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 2)


# ─── Main ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  STEP 1: Load Model from Hugging Face")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  โมเดล  : {MODEL_NAME}")
    print(f"  บันทึก : {MODEL_DIR}")
    print()

    # โหลด Feature Extractor + Model
    print("  กำลังโหลด feature extractor...")
    extractor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    extractor.save_pretrained(MODEL_DIR)

    print("  กำลังโหลดโมเดล (อาจใช้เวลาสักครู่)...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_DIR)
    model.eval()

    print(f"  Labels: {list(model.config.id2label.values())}")

    # ─── วัดขนาดโมเดล ───────────────────────────────────────
    size_mb = get_model_size_mb(MODEL_DIR)
    print(f"\n  ขนาดโมเดล (Original): {size_mb:.2f} MB")

    # ─── Step 2: Baseline Inference ─────────────────────────
    print()
    print("=" * 60)
    print("  STEP 2: Baseline Inference Test")
    print("=" * 60)

    image = get_test_image()
    inputs = extractor(images=image, return_tensors="pt")

    # Warmup
    print(f"  Warmup {NUM_WARMUP} ครั้ง...")
    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            _ = model(**inputs)

    # Measure
    print(f"  วัดค่า Latency {NUM_RUNS} ครั้ง...")
    latencies = []
    with torch.no_grad():
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            outputs = model(**inputs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    # ─── แสดงผล ─────────────────────────────────────────────
    mean_lat  = statistics.mean(latencies)
    p50_lat   = statistics.median(latencies)
    p95_lat   = sorted(latencies)[int(0.95 * NUM_RUNS)]
    min_lat   = min(latencies)

    # ดูผลลัพธ์ prediction
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class]

    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │        BASELINE RESULTS (Original)      │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Model size   : {size_mb:>8.2f} MB             │")
    print(f"  │  Mean latency : {mean_lat:>8.2f} ms             │")
    print(f"  │  P50 latency  : {p50_lat:>8.2f} ms             │")
    print(f"  │  P95 latency  : {p95_lat:>8.2f} ms             │")
    print(f"  │  Min latency  : {min_lat:>8.2f} ms             │")
    print(f"  │  Prediction   : {predicted_label:<24s} │")
    print("  └─────────────────────────────────────────┘")

    # บันทึกผลเพื่อนำไปเปรียบเทียบใน Step 4
    results = {
        "model": "Original (PyTorch)",
        "size_mb": round(size_mb, 2),
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "min_latency_ms": round(min_lat, 2),
    }

    import json
    results_path = Path("./models/benchmark_results.json")
    # โหลดผลเดิม (ถ้ามี) แล้ว append
    all_results = []
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
    # ลบผลเดิมของ Original ออก (ถ้ามี) แล้วใส่ใหม่
    all_results = [r for r in all_results if r["model"] != "Original (PyTorch)"]
    all_results.append(results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  บันทึกผลไปที่: {results_path}")


if __name__ == "__main__":
    main()
