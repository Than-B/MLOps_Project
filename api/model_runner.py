"""
model_runner.py
ทำงานใน child process (ProcessPoolExecutor)
- init_worker()   → โหลด ONNX session 1 ครั้งต่อ process
- predict_emotion() → รับ image bytes, คืน dict ผลลัพธ์

แยกไฟล์นี้ออกมาเพื่อหลีกเลี่ยง pickling ปัญหาของ ONNX session
"""

import io
import time
from typing import Dict, Any

import numpy as np
import onnxruntime as ort
from PIL import Image

# ─── Global state ต่อ process ─────────────────────────────
_session: ort.InferenceSession = None
_input_name: str = None
_id2label: Dict[int, str] = None

# Label map ของ dima806/facial_emotions_image_detection
EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

# Preprocessing constants (ViT-style, ตรงกับ Hugging Face preprocessor_config.json)
IMAGE_SIZE = 224
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def init_worker(model_path: str) -> None:
    """
    เรียกครั้งเดียวตอน process pool สร้าง worker
    โหลด ONNX session เข้า global variable ของ process นั้น
    """
    global _session, _input_name, _id2label

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # inter/intra thread ต่อ worker — ป้องกัน over-subscription
    sess_options.inter_op_num_threads = 1
    sess_options.intra_op_num_threads = 2

    _session = ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    _input_name = _session.get_inputs()[0].name
    _id2label = EMOTION_LABELS


def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    แปลง raw image bytes → numpy tensor [1, 3, 224, 224] float32
    ใช้ manual preprocessing แทน AutoFeatureExtractor เพื่อลด overhead
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)

    arr = np.array(img, dtype=np.float32) / 255.0          # [0,1]
    arr = (arr - MEAN) / STD                                 # normalize
    arr = arr.transpose(2, 0, 1)                             # HWC → CHW
    arr = np.expand_dims(arr, axis=0)                        # [1, 3, H, W]
    return arr.astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_emotion(image_bytes: bytes) -> Dict[str, Any]:
    """
    ฟังก์ชันหลักที่ถูกเรียกจาก main process ผ่าน executor
    รับ image bytes, คืน dict ผลลัพธ์
    """
    if _session is None:
        raise RuntimeError("Model session not initialized. init_worker() was not called.")

    # Preprocess
    input_tensor = preprocess(image_bytes)

    # Inference
    t0 = time.perf_counter()
    outputs = _session.run(None, {_input_name: input_tensor})
    t1 = time.perf_counter()

    inference_ms = (t1 - t0) * 1000

    # Postprocess
    logits = outputs[0][0]                          # shape: (num_classes,)
    probs  = softmax(logits)

    predicted_idx   = int(np.argmax(probs))
    predicted_label = _id2label[predicted_idx]
    confidence      = float(probs[predicted_idx])

    all_scores = {
        _id2label[i]: round(float(probs[i]), 4)
        for i in range(len(probs))
    }

    return {
        "predicted_emotion": predicted_label,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
        "inference_time_ms": round(inference_ms, 2),
    }
