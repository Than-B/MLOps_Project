"""
Facial Emotion Detection API
FastAPI + ONNX Runtime + ProcessPoolExecutor
"""

import asyncio
import io
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from schemas import PredictResponse, ErrorResponse, HealthResponse
from model_runner import predict_emotion, init_worker

# ─── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "5"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))   # ProcessPool workers
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).parent.parent / "models/onnx/model_quantized.onnx"),
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

# ─── Global process pool ──────────────────────────────────
executor: ProcessPoolExecutor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """startup / shutdown lifecycle"""
    global executor
    logger.info(f"Starting ProcessPoolExecutor with {NUM_WORKERS} workers...")
    logger.info(f"Model path: {MODEL_PATH}")

    if not Path(MODEL_PATH).exists():
        logger.error(f"Model not found at {MODEL_PATH}")
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # initializer โหลด ONNX session 1 ครั้งต่อ process (ไม่โหลดซ้ำทุก request)
    executor = ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_worker,
        initargs=(MODEL_PATH,),
    )
    logger.info("API ready ✓")
    yield
    # shutdown
    executor.shutdown(wait=True)
    logger.info("Executor shutdown complete")


# ─── App ──────────────────────────────────────────────────
app = FastAPI(
    title="Facial Emotion Detection API",
    description="Detect emotions from face images using optimized ONNX model",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware: log request timing ───────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
    return response


# ─── Exception handlers ───────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500},
    )


# ─── Validation helpers ───────────────────────────────────
def validate_upload(file: UploadFile, content: bytes) -> None:
    """ตรวจสอบ input ก่อนส่งเข้าโมเดล — raise HTTPException ถ้าไม่ผ่าน"""

    # 1. ขนาดไฟล์
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max allowed: {MAX_FILE_SIZE_MB} MB",
        )

    # 2. empty file
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # 3. content-type header
    content_type = file.content_type or ""
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type: '{content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )


def decode_image(content: bytes) -> Image.Image:
    """แปลง bytes → PIL Image พร้อมดักจับ corrupted file"""
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()  # ตรวจสอบว่าไม่ corrupt
        # verify() ปิด file pointer → เปิดใหม่เพื่อใช้งาน
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return img
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Cannot identify image. File may not be a valid image.",
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Image file appears to be corrupted or unreadable.",
        )


# ─── Routes ───────────────────────────────────────────────
@app.get("/", response_model=dict)
async def root():
    return {"message": "Facial Emotion Detection API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_path=MODEL_PATH,
        model_loaded=Path(MODEL_PATH).exists(),
        workers=NUM_WORKERS,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or corrupted file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Predict emotion from a face image",
)
async def predict(file: UploadFile = File(..., description="Face image (JPEG/PNG/WEBP)")):
    """
    รับภาพใบหน้าและคืนค่า emotion ที่ตรวจจับได้พร้อม confidence scores
    
    - รองรับ: JPEG, PNG, WEBP, GIF
    - ขนาดสูงสุด: 5 MB (ปรับได้ผ่าน MAX_FILE_SIZE_MB env)
    - ใช้ Quantized ONNX model (INT8) สำหรับ inference
    """
    # อ่าน bytes
    content = await file.read()

    # validate
    validate_upload(file, content)

    # decode image (ดักจับ corrupt/invalid)
    image = decode_image(content)

    # ส่งงาน CPU-bound ไปทำใน process pool (ไม่บล็อก event loop)
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(executor, predict_emotion, content)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    return PredictResponse(
        predicted_emotion=result["predicted_emotion"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        inference_time_ms=result["inference_time_ms"],
    )

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """รับหลายรูปพร้อมกัน คืนผลเป็น list"""
    results = []
    loop = asyncio.get_event_loop()

    for file in files:
        content = await file.read()
        try:
            validate_upload(file, content)
            decode_image(content)
            result = await loop.run_in_executor(executor, predict_emotion, content)
            results.append({"filename": file.filename, "status": "ok", **result})
        except HTTPException as e:
            results.append({"filename": file.filename, "status": "error", "error": e.detail})

    return {"total": len(results), "results": results}