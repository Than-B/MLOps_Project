import io
from fastapi.testclient import TestClient
from PIL import Image
from main import app

def create_dummy_image():
    """สร้างไฟล์รูปภาพสีทึบแบบหลอกๆ เอาไว้ใช้เทส"""
    img = Image.new("RGB", (224, 224), color=(128, 100, 90))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_predict_endpoint_success():
    """เทสว่า API ทำงานได้ ตอบกลับเป็น JSON และมีคีย์ครบถ้วน"""
    # ใช้ with เพื่อเปิดใช้งาน Lifespan ให้โหลดโมเดลก่อนเทส
    with TestClient(app) as client:
        img_bytes = create_dummy_image()
        response = client.post(
            "/predict",
            files={"file": ("test_image.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "class_id" in data
        assert "label" in data
        assert "confidence" in data

def test_predict_endpoint_invalid_file():
    """เทสการดักจับ Error กรณีอัปโหลดไฟล์ไม่ใช่รูปภาพ"""
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test_doc.txt", b"this is text", "text/plain")}
        )
        assert response.status_code == 400