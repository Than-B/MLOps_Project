import io
import os
import asyncio
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import onnxruntime as ort
from transformers import AutoImageProcessor
import numpy as np

# ==========================================
# ส่วนตั้งค่า (Configurations)
# ==========================================
MAX_FILE_SIZE = 5 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "onnx", "model_quantized.onnx")
PROCESSOR_DIR = os.path.join(BASE_DIR, "models", "original")

class PredictionResponse(BaseModel):
    class_id: int
    label: str
    confidence: float | None = None

# ==========================================
# Concurrency & Lifespan Management
# ==========================================
pool = None # ประกาศตัวแปร pool ไว้ก่อน

def init_worker():
    global session, processor, id2label
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    processor = AutoImageProcessor.from_pretrained(PROCESSOR_DIR)
    id2label = {0: 'sad', 1: 'disgust', 2: 'angry', 3: 'neutral', 4: 'fear', 5: 'surprise', 6: 'happy'}

# ใช้ Lifespan เพื่อคุมการเปิด/ปิด Process Pool ให้ปลอดภัยบน Windows
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    # Start: สร้าง Pool ตอนเริ่มเซิร์ฟเวอร์
    pool = ProcessPoolExecutor(max_workers=2, initializer=init_worker)
    yield
    # Stop: ปิด Pool ทิ้งตอนปิดเซิร์ฟเวอร์ ป้องกันแรมค้าง
    pool.shutdown(wait=True)

# ผูก lifespan เข้ากับ FastAPI
app = FastAPI(title="Facial Emotion Detection API", lifespan=lifespan)

# ==========================================
# Inference Logic
# ==========================================
def process_inference(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        inputs = processor(images=image, return_tensors="np")
        input_name = session.get_inputs()[0].name
        ort_input = {input_name: inputs["pixel_values"]}
        
        # รัน Model
        logits = session.run(None, ort_input)[0][0] # ดึง array มิติที่ 1 ออกมา
        
        # แปลงค่า Logits เป็น % Confidence ด้วยสมการ Softmax
        exp_logits = np.exp(logits - np.max(logits)) # ลบค่า max ป้องกัน overflow
        probabilities = exp_logits / exp_logits.sum()
        
        predicted_class = int(np.argmax(logits))
        confidence = float(probabilities[predicted_class]) * 100 # ทำเป็นเปอร์เซ็นต์
        
        return {
            "class_id": predicted_class, 
            "label": id2label.get(predicted_class, "unknown"),
            "confidence": round(confidence, 2) # ปัดเศษ 2 ตำแหน่ง
        }
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# API Endpoint
# ==========================================
@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
            detail=f"ขนาดไฟล์ใหญ่เกินไป (จำกัดที่ {MAX_FILE_SIZE / (1024*1024)} MB)"
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="ไฟล์ที่อัปโหลดต้องเป็นรูปภาพเท่านั้น (เช่น .jpg, .png)"
        )

    try:
        img_verify = Image.open(io.BytesIO(image_bytes))
        img_verify.verify() 
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="ไฟล์รูปภาพเสียหาย หรือฟอร์แมตไม่ถูกต้อง (Corrupted file)"
        )

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(pool, process_inference, image_bytes)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"เกิดข้อผิดพลาดในการรันโมเดล: {result['error']}"
        )

    return PredictionResponse(**result)