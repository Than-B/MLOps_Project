# ─── Stage 1: Builder ─────────────────────────────────────
# ติดตั้ง dependencies ใน layer แยก เพื่อให้ final image เล็กที่สุด
FROM python:3.11-slim AS builder

WORKDIR /build

# ติดตั้ง build tools ชั่วคราว
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements ก่อน (cache layer ถ้า requirements ไม่เปลี่ยน)
COPY api/requirements.txt .

# ติดตั้งลงใน prefix แยก เพื่อ copy ไป stage สุดท้าย
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ─── Stage 2: Production ──────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# รับค่าจาก build args (ปรับได้ตอน build)
ARG NUM_WORKERS=2
ARG MAX_FILE_SIZE_MB=5

ENV NUM_WORKERS=${NUM_WORKERS} \
    MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB} \
    MODEL_PATH=/app/models/onnx/model_quantized.onnx \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy Python packages จาก builder
COPY --from=builder /install /usr/local

# Copy โมเดล (quantized เท่านั้น — ไม่รวม original/onnx เพื่อลดขนาด)
COPY models/onnx/model_quantized.onnx /app/models/onnx/model_quantized.onnx

# Copy source code
COPY api/ /app/

# สร้าง non-root user (security best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

# Healthcheck (Hugging Face Spaces ใช้ port 7860)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# รัน uvicorn
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "info"]
