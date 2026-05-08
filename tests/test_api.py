"""
tests/test_api.py
Unit tests สำหรับ /predict endpoint
รัน: pytest tests/ -v
"""

import io
import asyncio
from unittest.mock import patch, AsyncMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ─── Mock result ──────────────────────────────────────────
MOCK_RESULT = {
    "predicted_emotion": "happy",
    "confidence": 0.923,
    "all_scores": {
        "angry": 0.012, "disgust": 0.003, "fear": 0.008,
        "happy": 0.923, "neutral": 0.041, "sad": 0.007, "surprise": 0.006,
    },
    "inference_time_ms": 45.2,
}


def make_image_bytes(color=(200, 150, 100), size=(224, 224), fmt="JPEG") -> bytes:
    """สร้าง dummy image bytes สำหรับทดสอบ"""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


# ─── Fixture ──────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    with patch("main.Path") as mock_path_cls, \
         patch("main.ProcessPoolExecutor"), \
         patch("main.init_worker"):

        mock_path_cls.return_value.exists.return_value = True
        mock_path_cls.return_value.__str__ = lambda self: "mock_model.onnx"

        from main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ─── Helper ───────────────────────────────────────────────
def mock_run_in_executor(result):
    async def _fake_run(executor, fn, *args):
        return result
    return _fake_run


# ─── Tests ────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "workers" in data


class TestPredictEndpoint:

    def _post_image(self, client, image_bytes, content_type="image/jpeg"):
        return client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, content_type)},
        )

    def test_predict_returns_200(self, client):
        img_bytes = make_image_bytes()
        with patch("main.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor(MOCK_RESULT)
            response = self._post_image(client, img_bytes)
        assert response.status_code == 200

    def test_predict_response_is_valid_json(self, client):
        img_bytes = make_image_bytes()
        with patch("main.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor(MOCK_RESULT)
            data = self._post_image(client, img_bytes).json()
        assert "predicted_emotion" in data
        assert "confidence" in data
        assert "all_scores" in data
        assert "inference_time_ms" in data

    def test_predict_confidence_in_range(self, client):
        img_bytes = make_image_bytes()
        with patch("main.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor(MOCK_RESULT)
            data = self._post_image(client, img_bytes).json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_all_scores_sum_approx_one(self, client):
        img_bytes = make_image_bytes()
        with patch("main.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor(MOCK_RESULT)
            data = self._post_image(client, img_bytes).json()
        total = sum(data["all_scores"].values())
        assert abs(total - 1.0) < 0.01

    def test_predict_emotion_label_valid(self, client):
        valid_emotions = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}
        img_bytes = make_image_bytes()
        with patch("main.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor(MOCK_RESULT)
            data = self._post_image(client, img_bytes).json()
        assert data["predicted_emotion"] in valid_emotions

    def test_predict_rejects_non_image_content_type(self, client):
        fake_pdf = b"%PDF-1.4 fake content"
        response = client.post(
            "/predict",
            files={"file": ("document.pdf", fake_pdf, "application/pdf")},
        )
        assert response.status_code == 415

    def test_predict_rejects_empty_file(self, client):
        response = self._post_image(client, b"")
        assert response.status_code == 400

    def test_predict_rejects_corrupted_image(self, client):
        corrupted = b"JFIF\x00\x00garbage_data_here_not_real_image"
        response = self._post_image(client, corrupted)
        assert response.status_code == 400

    def test_predict_rejects_oversized_file(self, client):
        big_bytes = b"\xff" * (6 * 1024 * 1024)
        response = self._post_image(client, big_bytes)
        assert response.status_code == 413

    def test_predict_requires_file(self, client):
        response = client.post("/predict")
        assert response.status_code == 422

    def test_predict_accepts_png(self, client):
        img_bytes = make_image_bytes(fmt="PNG")
        with patch("main.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = mock_run_in_executor(MOCK_RESULT)
            response = client.post(
                "/predict",
                files={"file": ("test.png", img_bytes, "image/png")},
            )
        assert response.status_code == 200
