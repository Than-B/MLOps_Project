"""
Step 4: Dynamic Quantization
แปลง ONNX → Quantized ONNX (INT8)
ลดขนาดและเพิ่มความเร็ว inference บน CPU
"""

import time
import json
import statistics
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference
from transformers import AutoImageProcessor
from PIL import Image


# ─── Config ───────────────────────────────────────────────
MODEL_DIR      = Path("./models/original")
ONNX_PATH      = Path("./models/onnx/model.onnx")
QUANTIZED_PATH = Path("./models/onnx/model_quantized.onnx")
NUM_WARMUP     = 3
NUM_RUNS       = 20


def get_model_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024 ** 2)
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 2)


def get_test_image() -> Image.Image:
    return Image.new("RGB", (224, 224), color=(128, 100, 90))


def main():
    QUANTIZED_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("     Dynamic Quantization (INT8)")
    print("=" * 60)

    if not ONNX_PATH.exists():
        print(f"  ERROR: ไม่พบ {ONNX_PATH}")
        print("  กรุณารัน 2_convert_onnx.py ก่อน")
        return

    # ─── Quantize ───────────────────────────────────────────
    PREPROCESSED_PATH = Path("./models/onnx/model_preprocessed.onnx") # เพิ่มไฟล์ทางผ่าน

    print(f"  กำลังทำ Pre-processing จัดระเบียบ Shape: {ONNX_PATH}")
    shape_inference.quant_pre_process(
        input_model_path=str(ONNX_PATH),
        output_model_path=str(PREPROCESSED_PATH),
        skip_optimization=False
    )

    print(f"  กำลัง quantize: {PREPROCESSED_PATH}")
    print(f"  บันทึกไปที่   : {QUANTIZED_PATH}")
    print()

    t0 = time.perf_counter()
    quantize_dynamic(
        model_input=str(PREPROCESSED_PATH), # เปลี่ยนมาใช้ไฟล์ที่ pre-process แล้ว
        model_output=str(QUANTIZED_PATH),
        weight_type=QuantType.QUInt8,
        # บังคับให้ใช้ external data ถ้าโมเดลมีขนาดใหญ่และแยก weight ไว้
        use_external_data_format=True 
    )
    t1 = time.perf_counter()
    print(f"  Quantization เสร็จใน {(t1 - t0):.2f} วินาที")

    # ─── เปรียบเทียบขนาด ────────────────────────────────────
    orig_size  = get_model_size_mb(Path("./models/original"))
    onnx_size  = get_model_size_mb(ONNX_PATH)
    quant_size = get_model_size_mb(QUANTIZED_PATH)

    print()
    print(f"  ขนาด Original : {orig_size:.2f} MB")
    print(f"  ขนาด ONNX     : {onnx_size:.2f} MB")
    print(f"  ขนาด Quantized: {quant_size:.2f} MB")
    print(f"  ลดจาก ONNX   : {((onnx_size - quant_size) / onnx_size * 100):.1f}%")

    # ─── Quantized Inference ────────────────────────────────
    print()
    print("  สร้าง Quantized Runtime session...")
    extractor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    image = get_test_image()
    inputs = extractor(images=image, return_tensors="pt")
    input_array = inputs["pixel_values"].numpy()

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(QUANTIZED_PATH), sess_options=sess_options,
                                   providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    ort_input = {input_name: input_array}

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

    logits = ort_outputs[0]
    predicted_class = int(np.argmax(logits, axis=-1)[0])

    print()
    print("  ┌─────────────────────────────────────────┐")
    print("  │      QUANTIZED (INT8) RESULTS           │")
    print("  ├─────────────────────────────────────────┤")
    print(f"  │  Model size   : {quant_size:>8.2f} MB              │")
    print(f"  │  Mean latency : {mean_lat:>8.2f} ms              │")
    print(f"  │  P50 latency  : {p50_lat:>8.2f} ms              │")
    print(f"  │  P95 latency  : {p95_lat:>8.2f} ms              │")
    print(f"  │  Min latency  : {min_lat:>8.2f} ms              │")
    print("  └─────────────────────────────────────────┘")

    # บันทึกผล
    results = {
        "model": "Quantized ONNX (INT8)",
        "size_mb": round(quant_size, 2),
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
    all_results = [r for r in all_results if r["model"] != "Quantized ONNX (INT8)"]
    all_results.append(results)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n  บันทึกผลไปที่: {results_path}")


if __name__ == "__main__":
    main()
