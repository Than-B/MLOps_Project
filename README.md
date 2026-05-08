# 🎭 Facial Emotion Detection API

ตรวจจับอารมณ์จากใบหน้า 7 ประเภท โดยใช้ Quantized ONNX model  
Model: [dima806/facial_emotions_image_detection](https://huggingface.co/dima806/facial_emotions_image_detection)

---

## โครงสร้างโปรเจค

```
project/
├── model_optimization/
│   ├── baseline_test.py       ← Step 1-2: baseline
│   ├── convert_onnx.py        ← Step 3: ONNX export
│   ├── quantize.py            ← Step 4: INT8 quantization
│   └── benchmark.py           ← Step 5: summary table
├── api/
│   ├── main.py                ← FastAPI app
│   ├── model_runner.py        ← ONNX inference (subprocess)
│   ├── schemas.py             ← Pydantic models
│   └── requirements.txt
├── tests/
│   ├── conftest.py
│   └── test_api.py            ← pytest unit tests
├── models/
│   └── onnx/
│       └── model_quantized.onnx
├── Dockerfile
└── .github/
    └── workflows/
        └── ci_cd.yml
```

---

## Emotions ที่ตรวจจับได้

| Label    | ความหมาย |
|----------|----------|
| angry    | โกรธ     |
| disgust  | รังเกียจ |
| fear     | กลัว     |
| happy    | มีความสุข |
| neutral  | เป็นกลาง |
| sad      | เศร้า    |
| surprise | แปลกใจ   |

---

## รันแบบ Local

### 1. ติดตั้ง + เตรียมโมเดล

```bash
pip install "numpy==1.26.4" torch torchvision transformers onnx onnxruntime Pillow

cd model_optimization
python baseline_test.py
python convert_onnx.py
python quantize.py
python benchmark.py
```

### 2. รัน API

```bash
pip install -r api/requirements.txt
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. ทดสอบ

```bash
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

---

## รันด้วย Docker

```bash
# Build
docker build -t emotion-api .

# Run
docker run -p 7860:7860 emotion-api

# Build พร้อมปรับ config
docker build --build-arg NUM_WORKERS=4 --build-arg MAX_FILE_SIZE_MB=10 -t emotion-api .
```

---

## API Endpoints

| Method | Path       | Description                  |
|--------|------------|------------------------------|
| GET    | `/`        | API info                     |
| GET    | `/health`  | Health check + model status  |
| POST   | `/predict` | Predict emotion from image   |
| GET    | `/docs`    | Swagger UI (interactive)     |

---

## cURL Examples

### Health Check

```bash
curl http://localhost:7860/health
```

### Predict (Local)

```bash
curl -X POST http://localhost:7860/predict \
  -F "file=MLOps\image\1.jpg" \
  -H "accept: application/json"
```

### Predict (Cloud — Hugging Face Spaces)

```bash
curl -X POST https://Than165-MLOps-api.hf.space/predict \
  -F "file=MLOps\image\1.jpg" \
  -H "accept: application/json"
```

### ตัวอย่าง Response

```json
{
  "predicted_emotion": "happy",
  "confidence": 0.9234,
  "all_scores": {
    "angry": 0.0121,
    "disgust": 0.0031,
    "fear": 0.0082,
    "happy": 0.9234,
    "neutral": 0.0412,
    "sad": 0.0071,
    "surprise": 0.0049
  },
  "inference_time_ms": 48.3
}
```

---

## HTTP Status Codes

| Code | เหตุ                                     |
|------|-----------------------------------------|
| 200  | สำเร็จ                                    |
| 400  | ไฟล์ไม่ใช่รูปภาพ / ไฟล์เสีย / ไฟล์เปล่า         |
| 413  | ไฟล์ใหญ่เกิน 5 MB                          |
| 415  | Content-Type ไม่ใช่รูปภาพ                  |
| 422  | ไม่ส่ง file field มา                      |
| 500  | Internal server error                   |

---

## Environment Variables

| Variable          | Default                            | Description              |
|-------------------|------------------------------------|--------------------------|
| `MODEL_PATH`      | `models/onnx/model_quantized.onnx` | Path to ONNX model       |
| `NUM_WORKERS`     | `2`                                | ProcessPool workers      |
| `MAX_FILE_SIZE_MB`| `5`                                | Max upload size (MB)     |

---

## CI/CD

GitHub Actions รัน 2 jobs:

1. **test** — รัน pytest ทุกครั้งที่ push (ทุก branch)
2. **deploy** — deploy ไป Hugging Face Spaces อัตโนมัติ **เฉพาะ** push ไป `main` และ test ผ่าน 100%

### GitHub Secrets ที่ต้องตั้งค่า

| Secret           | ค่า                           |
|------------------|------------------------------|
| `HF_TOKEN`       | Hugging Face Access Token    |
| `HF_USERNAME`    | HF username                  |
| `HF_SPACE_NAME`  | ชื่อ Space บน Hugging Face     |


#Apache Setting

##Huggingface
protocal: https
server: Than165-MLOps-api.hf.space
port: 443

##Docker
protocal: http
server: localhost
port: 7860