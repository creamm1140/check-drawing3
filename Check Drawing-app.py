import streamlit as st
import io
import json
import time
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
from google import genai
from google.genai import types

# ----------------- ตั้งค่าหน้าเว็บ Streamlit -----------------
st.set_page_config(
    page_title="AI ตรวจแบบแปลนวิศวกรรมชลประทาน",
    page_icon="🚜",
    layout="wide"
)

st.title("🚜 AI ตรวจสอบแบบแปลนวิศวกรรมชลประทาน")
st.caption("ระบบตรวจสอบมิติโครงสร้าง, ระดับชลศาสตร์, สัญลักษณ์ และคำผิดอัตโนมัติ (Powered by Gemini API)")

# ----------------- เมนูแถบด้านข้าง (Sidebar) -----------------
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Gemini API Key:", value=default_key, type="password")
    
    selected_model = st.selectbox(
        "เลือกรุ่นโมเดล AI:",
        ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        index=0,
        help="หากตัวหลักเกิดความหนาแน่น ให้สลับไปใช้รุ่น lite แทนได้"
    )
    st.markdown("[👉 สมัครรับ API Key ฟรีที่ Google AI Studio](https://aistudio.google.com/)")

if not api_key:
    st.warning("⚠️ กรุณากรอก Gemini API Key ในแถบด้านซ้ายก่อนเริ่มใช้งาน")
    st.stop()

# เชื่อมต่อ Client
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini API: {e}")
    st.stop()

# ----------------- Prompt ตรวจสอบวิศวกรรม -----------------
PROMPT = """
คุณคือวิศวกรผู้เชี่ยวชาญด้านการตรวจสอบแบบแปลนวิศวกรรมชลประทานและโครงสร้างอาคารชลประทาน
จงตรวจสอบแบบแปลนในภาพนี้อย่างละเอียดใน 4 มิติ:
1. มิติโครงสร้าง (Dimensions & Scale): ตัวเลขระยะ ความสูง ความกว้าง ขัดแย้งหรือไม่
2. หลักวิศวกรรมชลประทาน (Hydraulics & Levels): ระดับน้ำ (ร.น.ส., ร.ม.ส.), ทิศทางการไหล, สโลป, สันฝาย, บานระบาย, ท่อลอด, ทรบ.
3. โครงสร้างและวัสดุ: ระยะเหล็ก, รอยต่อ, สัญลักษณ์
4. คำผิด/คำตกหล่น (ทั้งภาษาไทยและอังกฤษ)

เงื่อนไข: ตอบกลับเป็น JSON Array เท่านั้น ในรูปแบบ:
[
  {
    "box_2d": [ymin, xmin, ymax, xmax],
    "type": "error" หรือ "observation",
    "title": "หัวข้อจุดที่พบ",
    "description": "คำอธิบายข้อผิดพลาดหรือข้อสังเกตเชิงวิศวกรรม"
  }
]
* หมายเหตุ: box_2d ต้องเป็นพิกัดสัดส่วน 0 ถึง 1000
"""

# ----------------- ฟังก์ชันประมวลผล -----------------
def analyze_with_retry(client, model, contents, config, max_retries=4, initial_delay=3):
    """ส่งตรวจพร้อมระบบ Auto-Retry เมื่อเจอ Error 503 หรือ 429"""
    delay = initial_delay
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            # ตัด Markdown Code Block ออกหากมีติดมา
            text_resp = response.text.strip()
            if text_resp.startswith("```json"):
                text_resp = text_resp[7:]
            if text_resp.endswith("```"):
                text_resp = text_resp[:-3]
            return json.loads(text_resp.strip())
        except Exception as e:
            last_error = e
            err_msg = str(e)
            if any(code in err_msg for code in ["503", "429", "UNAVAILABLE", "high demand"]):
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 12  # ทวีคูณเวลารอ 3s -> 6s -> 12s
                    continue
            raise last_error
    raise last_error

def annotate_image(image: Image.Image, findings: list) -> Image.Image:
    """วาดกรอบสีแดง/ส้มและป้ายกำกับลงบนภาพ"""
    draw_img = image.copy().convert("RGB")
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

            # 1. วาดกรอบสี่เหลี่ยมรอบจุดที่พบ
            draw.rectangle([left, top, right, bottom], outline=color, width=5)
            
            # 2. ป้ายข้อความกำกับ
            tag = f"#{idx} {item.get('title', '')}"
            draw.rectangle([left, max(0, top - 25), left + max(len(tag) * 11, 70), top], fill=color)
            draw.text((left + 5, max(0, top - 20)), tag, fill="white")

    return draw_img

# ----------------- ส่วน UI การอัปโหลด -----------------
uploaded_file = st.file_uploader(
    "📁 อัปโหลดแบบแปลน (รองรับ PDF, JPG, PNG, TIFF):", 
    type=["pdf", "jpg", "jpeg", "png", "tiff"]
)

if uploaded_file and st.button("🚀 เริ่มการตรวจสอบ (Start Audit)"):
    file_bytes = uploaded_file.read()
    raw_pages = []

    with st.spinner("กำลังแปลงไฟล์เอกสารเป็นรูปภาพ..."):
        if uploaded_file.name.lower().endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                raw_pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        else:
            raw_pages.append(Image.open(io.BytesIO(file_bytes)).convert("RGB"))

    st.info(f"ตรวจพบทั้งหมด {len(raw_pages)} หน้า กำลังส่งให้ AI ทำการตรวจสอบ...")

    annotated_pages = []
    all_findings = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, page_img in enumerate(raw_pages):
        status_text.text(f"⏳ กำลังตรวจสอบหน้าที่ {idx + 1}/{len(raw_pages)}...")
        
        img_buffer = io.BytesIO()
        page_img.save(img_buffer, format="JPEG", quality=85)
        
        findings = []
        try:
            findings = analyze_with_retry(
                client=client,
                model=selected_model,
                contents=[
                    types.Part.from_bytes(data=img_buffer.getvalue(), mime_type="image/jpeg"),
                    PROMPT
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
                max_retries=4,
                initial_delay=3
            )
        except Exception as e:
            st.warning(f"⚠️ หน้าที่ {idx + 1} ไม่สามารถตรวจสอบได้เนื่องจากปัญหาการเชื่อมต่อ: {e}")
            findings = []

        annotated_img = annotate_image(page_img, findings)
        annotated_pages.append(annotated_img)
        all_findings.append(findings)

        progress_bar.progress((idx + 1) / len(raw_pages))
        time.sleep(2.0)  # เว้นระยะเพื่อป้องกัน Rate Limit ของ Free Tier

    status_text.empty()
    st.success("✅ ตรวจสอบแบบแปลนเสร็จสิ้นเรียบร้อยแล้ว!")

    # ----------------- แสดงผลการตรวจสอบทีละหน้า -----------------
    st.subheader("📋 รายละเอียดผลการตรวจสอบ")
    for i, (ann_img, page_finds) in enumerate(zip(annotated_pages, all_findings)):
        with st.expander(f"📄 หน้าที่ {i + 1} (พบ {len(page_finds)} รายการ)", expanded=(i == 0)):
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.image(ann_img, use_container_width=True, caption=f"ภาพผลการตรวจหน้าที่ {i + 1}")
            
            with col2:
                st.markdown("#### ข้อสังเกตและจุดผิดพลาด:")
                if not page_finds:
                    st.write("🟢 ไม่พบข้อผิดพลาดที่ชัดเจนในหน้านี้")
                for f_idx, item in enumerate(page_finds, start=1):
                    badge = "🔴 ผิดพลาด (Error)" if item.get("type") == "error" else "🟠 ข้อสังเกต (Observation)"
                    st.markdown(f"**#{f_idx} [{badge}] {item.get('title')}**")
                    st.write(item.get("description"))
                    st.divider()

    # ----------------- ปุ่มดาวน์โหลด PDF ผลการตรวจสอบ -----------------
    output_pdf = io.BytesIO()
    if annotated_pages:
        # บันทึกเป็น PDF รวมทุกหน้า
        annotated_pages[0].save(
            output_pdf,
            format="PDF",
            save_all=True,
            append_images=annotated_pages[1:]
        )
        output_pdf.seek(0)

        st.download_button(
            label="📥 ดาวน์โหลดแบบแปลนที่มีการวงจุดผิด (Annotated PDF)",
            data=output_pdf,
            file_name=f"Audited_{uploaded_file.name}.pdf",
            mime="application/pdf"
        )
