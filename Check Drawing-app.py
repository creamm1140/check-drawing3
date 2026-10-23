import streamlit as st
import io
import json
import time
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI ตรวจแบบแปลนชลประทาน", layout="wide")

st.title("🚜 AI ตรวจสอบแบบแปลนวิศวกรรมชลประทาน")
st.caption("ระบบตรวจสอบมิติโครงสร้าง, ชลศาสตร์, สัญลักษณ์ และคำผิด อัตโนมัติ")

# ----------------- ส่วนจัดการแถบด้านข้าง (Sidebar) -----------------
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Gemini API Key:", value=default_key, type="password")
    
    # เลือกรุ่นโมเดล (ใช้ gemini-3.8-flash เป็นค่าเริ่มต้น)
    selected_model = st.selectbox(
        "เลือกรุ่นโมเดล AI:",
        ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        index=0,
        help="gemini-3.8-flash เป็นโมเดลที่แนะนำและรองรับการตรวจแบบแปลนความละเอียดสูงได้ดีที่สุด"
    )
    st.markdown("[👉 สมัครรับ API Key ฟรีที่นี่](https://aistudio.google.com/)")

if not api_key:
    st.warning("⚠️ กรุณากรอก Gemini API Key ในแถบด้านซ้ายก่อนเริ่มใช้งาน")
    st.stop()

# สร้าง Client
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini API: {e}")
    st.stop()

# ----------------- ฟังก์ชันช่วยประมวลผล -----------------
PROMPT = """
คุณคือวิศวกรผู้เชี่ยวชาญด้านการตรวจสอบแบบแปลนวิศวกรรมชลประทานและโครงสร้างอาคารชลประทาน
จงตรวจสอบแบบแปลนในภาพนี้อย่างละเอียด:
1. มิติโครงสร้าง (Dimensions): ตัวเลขระยะ ความสูง ความกว้าง ขัดแย้งหรือไม่
2. หลักวิศวกรรมชลประทาน (Hydraulics): ระดับน้ำ (ร.น.ส., ร.ม.ส.), ทิศทางการไหล, สโลป, สันฝาย, ท่อลอด, ทรบ.
3. โครงสร้างและวัสดุ: ระยะเหล็ก, รอยต่อ, สัญลักษณ์
4. คำผิด/คำตกหล่น (ภาษาไทยและอังกฤษ)

ตอบกลับเป็น JSON Array เท่านั้น ในรูปแบบ:
[
  {
    "box_2d": [ymin, xmin, ymax, xmax],
    "type": "error" หรือ "observation",
    "title": "หัวข้อจุดที่พบ",
    "description": "คำอธิบายข้อผิดพลาดหรือข้อสังเกต"
  }
]
* หมายเหตุ: box_2d ต้องเป็นพิกัดสัดส่วน 0 ถึง 1000
"""

def annotate_image(image: Image.Image, findings: list) -> Image.Image:
    """วาดกรอบสีแดง/ส้มและโน้ตลงบนภาพ"""
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    w, h = draw_img.size

    for idx, item in enumerate(findings, start=1):
        box = item.get("box_2d", [])
        if len(box) == 4:
            ymin, xmin, ymax, xmax = box
            left = (xmin / 1000) * w
            top = (ymin / 1000) * h
            right = (xmax / 1000) * w
            bottom = (ymax / 1000) * h

            is_error = item.get("type") == "error"
            color = "red" if is_error else "orange"

            # วาดกรอบสี่เหลี่ยมเน้นจุดผิด
            draw.rectangle([left, top, right, bottom], outline=color, width=5)
            
            # วาดป้ายข้อความกำกับ
            tag = f"#{idx} {item.get('title', '')}"
            draw.rectangle([left, max(0, top - 25), left + len(tag) * 12, top], fill=color)
            draw.text((left + 5, max(0, top - 20)), tag, fill="white")

    return draw_img

# ----------------- ส่วน UI อัปโหลด -----------------
uploaded_file = st.file_uploader(
    "📁 อัปโหลดแบบแปลน (รองรับ PDF, JPG, PNG, TIFF):", 
    type=["pdf", "jpg", "jpeg", "png", "tiff"]
)

if uploaded_file and st.button("🚀 เริ่มการตรวจสอบ (Start Audit)"):
    file_bytes = uploaded_file.read()
    raw_pages = []

    with st.spinner("กำลังแปลงไฟล์เป็นรูปภาพ..."):
        if uploaded_file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                raw_pages.append(Image.open(io.BytesIO(pix.tobytes("png"))))
        else:
            raw_pages.append(Image.open(io.BytesIO(file_bytes)).convert("RGB"))

    st.info(f"ตรวจพบทั้งหมด {len(raw_pages)} หน้า กำลังส่งให้ AI ตรวจสอบทีละหน้า...")

    annotated_pages = []
    all_findings = []

    progress_bar = st.progress(0)
    for idx, page_img in enumerate(raw_pages):
        img_buffer = io.BytesIO()
        page_img.save(img_buffer, format="JPEG", quality=90)
        
        try:
            # เรียกใช้โมเดลตามที่เลือกจาก Sidebar
            response = client.models.generate_content(
                model=selected_model,
                contents=[
                    types.Part.from_bytes(data=img_buffer.getvalue(), mime_type="image/jpeg"),
                    PROMPT
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            findings = json.loads(response.text)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในหน้าที่ {idx + 1}: {e}")
            findings = []

        annotated_img = annotate_image(page_img, findings)
        annotated_pages.append(annotated_img)
        all_findings.append(findings)

        progress_bar.progress((idx + 1) / len(raw_pages))
        time.sleep(1)

    st.success("✅ ตรวจสอบแบบแปลนเสร็จสิ้น!")

    # ----------------- แสดงผลลัพธ์ทีละหน้า -----------------
    st.subheader("📋 ผลการตรวจสอบแต่ละหน้า")
    for i, (ann_img, page_finds) in enumerate(zip(annotated_pages, all_findings)):
        with st.expander(f"📄 หน้าที่ {i + 1} (พบ {len(page_finds)} รายการ)", expanded=(i == 0)):
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.image(ann_img, use_container_width=True, caption=f"ภาพผลการตรวจหน้าที่ {i + 1}")
            
            with col2:
                st.markdown("#### รายการข้อสังเกต/จุดผิดพลาด:")
                if not page_finds:
                    st.write("🟢 ไม่พบข้อผิดพลาดที่ชัดเจนในหน้านี้")
                for f_idx, item in enumerate(page_finds, start=1):
                    badge = "🔴 ผิดพลาด (Error)" if item.get("type") == "error" else "🟠 ข้อสังเกต (Observation)"
                    st.markdown(f"**#{f_idx} [{badge}] {item.get('title')}**")
                    st.write(item.get("description"))
                    st.divider()

    # ----------------- ปุ่มดาวน์โหลด PDF รวมผล -----------------
    output_pdf = io.BytesIO()
    if annotated_pages:
        annotated_pages[0].save(
            output_pdf, format="PDF", save_all=True, append_images=annotated_pages[1:]
        )
        output_pdf.seek(0)

        st.download_button(
            label="📥 ดาวน์โหลดไฟล์แบบแปลนที่มีการวงจุดผิด (Annotated PDF)",
            data=output_pdf,
            file_name=f"Audited_{uploaded_file.name}.pdf",
            mime="application/pdf"
        )
