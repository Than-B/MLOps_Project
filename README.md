---
title: MLOps Emotion API
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Phase 1: Model Optimization
(เนื้อหาเดิมของคุณต่อลงมาด้านล่าง...)

## โมเดลที่ใช้
**dima806/facial_emotions_image_detection**  
ตรวจจับอารมณ์จากใบหน้า 7 ประเภท: angry, disgust, fear, happy, neutral, sad, surprise

---

## ติดตั้ง Dependencies

```bash
pip install "numpy==1.26.4" torch torchvision transformers onnx onnxruntime Pillow requests
```

หรือถ้าใช้ virtual environment (แนะนำ):
```bash
python -m venv venv
venv\Scripts\activate
pip install "numpy==1.26.4" torch torchvision transformers onnx onnxruntime Pillow requests
```

---

## ขั้นตอนการรัน

### 1. Baseline Test
```bash
python baseline_test.py
```

### 2. แปลงเป็น ONNX
```bash
python convert_onnx.py
```

### 3. Dynamic Quantization
```bash
python quantize.py
```

### 4. ดูตารางสรุป
```bash
python benchmark.py
```

---

## ผลที่คาดหวัง

| โมเดล               | Size (MB) | Mean (ms) | P95 (ms) | vs Original |
|---------------------|-----------|-----------|----------|-------------|
| Original (PyTorch)  | ~100 MB   | ~150 ms   | ~180 ms  | baseline    |
| ONNX                | ~100 MB   | ~80 ms    | ~100 ms  | -47%        |
| Quantized ONNX INT8 | ~25 MB    | ~50 ms    | ~65 ms   | -67%        |

---

## โครงสร้างไฟล์หลังรันครบ

```
models/
├── original/
├── onnx/
│   ├── model.onnx
│   └── model_quantized.onnx   ← ใช้ใน Production
└── benchmark_results.json
```