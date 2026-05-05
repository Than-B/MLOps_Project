# ใช้ image แบบ slim เพื่อให้ขนาดเล็กที่สุด (ลบเครื่องมือที่ไม่จำเป็นออก)
FROM python:3.11-slim

# ตั้งค่า Environment ลดการเขียนไฟล์ขยะของ Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ก๊อปปี้ไฟล์ requirements มาก่อนเพื่อทำ Caching (ช่วยให้บิลด์รอบหน้าเร็วขึ้น)
COPY requirements.txt .

# ติดตั้งไลบรารีแบบไม่เก็บ Cache ไว้ใน Image (ลดขนาดไฟล์)
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้ไฟล์โค้ดและโมเดลทั้งหมดเข้า Image
COPY . .

# เปิด Port 8000
EXPOSE 8000

# รัน Uvicorn server แบบ Single Worker 
# (เพราะเรามีการจัดการ ProcessPool แยกไว้ข้างในโค้ดแล้ว)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]