"""
Pydantic Schemas — Request / Response validation
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictResponse(BaseModel):
    predicted_emotion: str = Field(..., description="Emotion with highest confidence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    all_scores: Dict[str, float] = Field(..., description="Score for every emotion class")
    inference_time_ms: float = Field(..., description="Model inference time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_emotion": "happy",
                "confidence": 0.923,
                "all_scores": {
                    "angry": 0.012,
                    "disgust": 0.003,
                    "fear": 0.008,
                    "happy": 0.923,
                    "neutral": 0.041,
                    "sad": 0.007,
                    "surprise": 0.006,
                },
                "inference_time_ms": 48.3,
            }
        }
    }


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "File too large. Max allowed: 5 MB",
                "status_code": 413,
            }
        }
    }


class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    model_path: str = Field(..., description="Path to loaded ONNX model")
    model_loaded: bool = Field(..., description="Whether model file exists")
    workers: int = Field(..., description="Number of process pool workers")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "model_path": "models/onnx/model_quantized.onnx",
                "model_loaded": True,
                "workers": 2,
            }
        }
    }
