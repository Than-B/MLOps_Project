"""
Step 5: Benchmark Summary
อ่านผลจาก benchmark_results.json และแสดงตารางเปรียบเทียบ
Original vs ONNX vs Quantized
"""

import json
from pathlib import Path


RESULTS_PATH = Path("./models/benchmark_results.json")

ORDER = ["Original (PyTorch)", "ONNX", "Quantized ONNX (INT8)"]


def pct_change(baseline: float, new: float) -> str:
    if baseline == 0:
        return "N/A"
    pct = (new - baseline) / baseline * 100
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.1f}%"


def main():
    if not RESULTS_PATH.exists():
        print("ไม่พบไฟล์ผล กรุณารัน 1_baseline_test.py, 2_convert_onnx.py, 3_quantize.py ก่อน")
        return

    with open(RESULTS_PATH) as f:
        all_results: list[dict] = json.load(f)

    # จัดเรียงตาม ORDER
    result_map = {r["model"]: r for r in all_results}
    rows = [result_map[k] for k in ORDER if k in result_map]

    if not rows:
        print("ไม่มีข้อมูล")
        return

    baseline = rows[0]

    print()
    print("=" * 72)
    print("  MODEL OPTIMIZATION BENCHMARK — SUMMARY TABLE")
    print("=" * 72)
    print(f"  {'โมเดล':<28} {'Size (MB)':>10} {'Mean (ms)':>10} {'P95 (ms)':>10} {'vs Original':>12}")
    print("  " + "-" * 70)

    for r in rows:
        name     = r["model"]
        size     = r["size_mb"]
        mean_lat = r["mean_latency_ms"]
        p95_lat  = r["p95_latency_ms"]
        vs_orig  = pct_change(baseline["mean_latency_ms"], mean_lat) if name != baseline["model"] else "baseline"
        print(f"  {name:<28} {size:>10.2f} {mean_lat:>10.2f} {p95_lat:>10.2f} {vs_orig:>12}")

    print("  " + "-" * 70)

    if len(rows) == 3:
        orig_size  = rows[0]["size_mb"]
        quant_size = rows[2]["size_mb"]
        orig_lat   = rows[0]["mean_latency_ms"]
        quant_lat  = rows[2]["mean_latency_ms"]

        size_reduction = (orig_size - quant_size) / orig_size * 100
        speed_improvement = (orig_lat - quant_lat) / orig_lat * 100

        print()
        print(f"  สรุป: Quantized model ลดขนาดได้ {size_reduction:.1f}%  "
              f"| เร็วขึ้น {speed_improvement:.1f}% (mean latency)")

    print()
    print("  ไฟล์ที่สำคัญ:")
    print("    models/original/           → PyTorch weights")
    print("    models/onnx/model.onnx     → ONNX format")
    print("    models/onnx/model_quantized.onnx → ใช้สำหรับ Production API")
    print()
    print("=" * 72)


if __name__ == "__main__":
    main()
