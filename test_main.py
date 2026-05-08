import io
from fastapi.testclient import TestClient
from PIL import Image
from main import app

# สร้าง Client จำลองสำหรับทดสอบ API โดยไม่ต้องเปิด Server จริง
client = TestClient(app)

def create_dummy_image():
    """สร้างไฟล์รูปภาพสีทึบแบบหลอกๆ เอาไว้ใช้เทส"""
    img = Image.new("RGB", (224, 224), color=(128, 100, 90))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def test_predict_endpoint_success():
    """เทสว่า API ทำงานได้ ตอบกลับเป็น JSON และมีคีย์ครบถ้วน"""
    img_bytes = create_dummy_image()
    
    # จำลองการยิง POST Request พร้อมแนบไฟล์
    response = client.post(
        "/predict",
        files={"file": ("test_image.jpg", img_bytes, "image/jpeg")}
    )
    
    # 1. เช็ค Status Code ว่าสำเร็จ (200 OK)
    assert response.status_code == 200
    
    # 2. เช็คว่าเป็น JSON รูปแบบที่ถูกต้อง
    data = response.json()
    assert "class_id" in data
    assert "label" in data
    assert "confidence" in data
    
    # 3. เช็คว่าโมเดลทำนายได้ (ไม่พ่น None ออกมา)
    assert isinstance(data["label"], str)
    assert isinstance(data["confidence"], float)

def test_predict_endpoint_invalid_file():
    """เทสการดักจับ Error กรณีอัปโหลดไฟล์ไม่ใช่รูปภาพ"""
    response = client.post(
        "/predict",
        files={"file": ("test_doc.txt", b"this is text", "text/plain")}
    )
    
    # ต้องตอบกลับเป็น 400 Bad Request
    assert response.status_code == 400