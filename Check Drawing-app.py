import io
import json
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

app = FastAPI()

# สมัครรับ API Key ฟรีได้ที่ https://aistudio.google.com/
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

PROMPT = """
คุณคือวิศวกรชลประทานผู้เชี่ยวชาญ ตรวจสอบแบบแปลนนี้อย่างละเอียด:
1. มิติโครงสร้าง (Dimensions & Scale)
2. สัญลักษณ์และความถูกต้องตามหลักชลศาสตร์ (Hydraulics & Levels)
3. คำผิด/ตัวสะกด (ภาษาไทยและอังกฤษ)

ส่งผลลัพธ์เป็น JSON Array เท่านั้น ในรูปแบบ:
[
  {
    "box_2d": [ymin, xmin, ymax, xmax],
    "type": "error" หรือ "observation",
    "title": "หัวข้อ",
    "description": "คำอธิบาย"
  }
]
พิกัด box_2d ใช้สเกล 0 ถึง 1000
"""

def annotate_image(image: Image.Image, findings: list) -> Image.Image:
    """วาดกรอบสีแดงและข้อความโน้ตลงบนภาพ"""
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for idx, item in enumerate(findings, start=1):
        box = item.get("box_2d", [])
        if len(box) == 4:
            ymin, xmin, ymax, xmax = box
            # แปลงสัดส่วน 0-1000 เป็นพิกัดจริงบนภาพ
            left = (xmin / 1000) * width
            top = (ymin / 1000) * height
            right = (xmax / 1000) * width
            bottom = (ymax / 1000) * height

            color = "red" if item.get("type") == "error" else "orange"

            # วาดกรอบสี่เหลี่ยม/วงกลมเน้นจุดผิด
            draw.rectangle([left, top, right, bottom], outline=color, width=4)
            
            # วาดป้ายข้อสังเกต (Badge & Text)
            label = f"#{idx} [{item.get('title')}]: {item.get('description')}"
            draw.rectangle([left, max(0, top - 25), left + len(label) * 8, top], fill=color)
            draw.text((left + 5, max(0, top - 20)), label, fill="white")

    return image

@app.post("/audit-drawing/")
async def audit_drawing(file: UploadFile = File(...)):
    file_bytes = await file.read()
    annotated_images = []

    # 1. แปลงไฟล์เข้าสู่รูปภาพแต่ละหน้า (รองรับทั้ง PDF, PNG, JPG)
    if file.filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=200) # ปรับ DPI ให้คมชัดพอที่ AI จะอ่านแบบได้
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            pages.append(img)
    else:
        pages = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

    # 2. ให้ AI ตรวจทีละหน้า
    for page_img in pages:
        # บีบอัดชั่วคราวเพื่อส่งเข้า Gemini
        img_byte_arr = io.BytesIO()
        page_img.save(img_byte_arr, format='JPEG')
        
        response = client.models.generate_content(
            model='gemini-2.0-flash', # หรือ gemini-1.5-flash (ฟรี)
            contents=[
                types.Part.from_bytes(data=img_byte_arr.getvalue(), mime_type='image/jpeg'),
                PROMPT
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        try:
            findings = json.loads(response.text)
        except Exception:
            findings = []

        # 3. วาดผลการตรวจสอบลงบนหน้าแบบแปลน
        annotated_img = annotate_image(page_img.copy(), findings)
        annotated_images.append(annotated_img)

    # 4. รวมหน้าทั้งหมดกลับเป็น PDF ส่งให้ดาวน์โหลด
    output_pdf = io.BytesIO()
    if annotated_images:
        first_img = annotated_images[0]
        rest_imgs = annotated_images[1:]
        first_img.save(output_pdf, format="PDF", save_all=True, append_images=rest_imgs)
        output_pdf.seek(0)

    return StreamingResponse(
        output_pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=Audited_{file.filename}.pdf"}
    )
