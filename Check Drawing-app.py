import streamlit as st
import io
import json
import time
import os
import urllib.request
from datetime import datetime
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

# ----------------- ตั้งค่าหน้าเว็บ -----------------
st.set_page_config(
    page_title="AI ตรวจแบบแปลนวิศวกรรมชลประทาน",
    page_icon="🚜",
    layout="wide"
)

st.title("🚜 AI ตรวจสอบแบบแปลนวิศวกรรมชลประทาน")
st.caption("ระบบตรวจสอบมิติโครงสร้าง, ชลศาสตร์, สัญลักษณ์ และคำผิด (รองรับภาษาไทย 100%)")

# ----------------- ระบบดาวน์โหลดและโหลดฟอนต์ภาษาไทย -----------------
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
FONT_PATH = "Sarabun-Regular.ttf"

@st.cache_resource
def load_thai_font(size: int = 20):
    """ดาวน์โหลดฟอนต์ Sarabun อัตโนมัติหากยังไม่มีในระบบ เพื่อแสดงผลภาษาไทยได้ถูกต้อง"""
    if not os.path.exists(FONT_PATH):
        try:
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        except Exception:
            pass

    # 1. พยายามโหลดฟอนต์ Sarabun
    if os.path.exists(FONT_PATH):
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except Exception:
            pass

    # 2. สำรองสำหรับ Windows (Tahoma)
    for win_font in ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/angsana.ttf"]:
        if os.path.exists(win_font):
            try:
                return ImageFont.truetype(win_font, size)
            except Exception:
                pass

    # 3. สำรองสำหรับ Mac (Thonburi)
    for mac_font in ["/System/Library/Fonts/Supplemental/Thonburi.ttc", "/Library/Fonts/Thonburi.ttc"]:
        if os.path.exists(mac_font):
            try:
                return ImageFont.truetype(mac_font, size)
            except Exception:
                pass

    return ImageFont.load_default()

def wrap_text_thai(text: str, font, max_width: int, draw: ImageDraw.Draw) -> list:
    """ฟังก์ชันตัดบรรทัดภาษาไทยตามความกว้างพิกเซลจริง ป้องกันข้อความล้นกระดาษ"""
    lines = []
    current_line = ""
    for char in text:
        test_line = current_line + char
        # คำนวณความกว้างตัวอักษรจริง
        if draw.textlength(test_line, font=font) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = char
    if current_line:
        lines.append(current_line)
    return lines

# ----------------- เมนูแถบด้านข้าง (Sidebar) -----------------
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Gemini API Key:", value=default_key, type="password")
    
    selected_model = st.selectbox(
        "เลือกรุ่นโมเดล AI:",
        ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        index=0,
        help="หากตัวหลักติดโควตารายวัน ระบบจะสลับไปใช้รุ่น lite สำรองให้อัตโนมัติ"
    )
    st.markdown("[👉 สมัครรับ API Key ฟรีที่ Google AI Studio](https://aistudio.google.com/)")

if not api_key:
    st.warning("⚠️ กรุณากรอก Gemini API Key ในแถบด้านซ้ายก่อนเริ่มใช้งาน")
    st.stop()

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini API: {e}")
    st.stop()

# ----------------- Prompt ตรวจสอบ -----------------
PROMPT = """
คุณคือวิศวกรผู้เชี่ยวชาญด้านการตรวจสอบแบบแปลนวิศวกรรมชลประทานและโครงสร้างอาคารชลประทาน
จงตรวจสอบแบบแปลนในภาพนี้อย่างละเอียดใน 4 มิติ:
1. มิติโครงสร้าง (Dimensions & Scale): ตัวเลขระยะ ความสูง ความกว้าง ขัดแย้งหรือไม่
2. หลักวิศวกรรมชลประทาน (Hydraulics & Levels): ระดับน้ำ (ร.น.ส., ร.ม.ส.), ทิศทางการไหล, สโลป, สันฝาย, บานระบาย, ท่อลอด, ทรบ.
3. โครงสร้างและวัสดุ: ระยะเหล็ก, รอยต่อ, สัญลักษณ์
4. คำผิด/คำตกหล่น (ภาษาไทยและอังกฤษ)

ตอบกลับเป็น JSON Array เท่านั้น ในรูปแบบ:
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

def analyze_with_fallback(client, primary_model, contents, config, fallback_model="gemini-3.5-flash-lite"):
    """ระบบตรวจพร้อม Auto-Retry และ Fallback"""
    models_to_try = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models_to_try.append(fallback_model)

    last_error = None
    for model_name in models_to_try:
        for _ in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                text_resp = response.text.strip()
                if text_resp.startswith("```json"):
                    text_resp = text_resp[7:]
                if text_resp.endswith("```"):
                    text_resp = text_resp[:-3]
                return json.loads(text_resp.strip()), model_name
            except Exception as e:
                last_error = e
                err_msg = str(e)
                if "503" in err_msg or "UNAVAILABLE" in err_msg:
                    time.sleep(3)
                    continue
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    break
    raise last_error

def annotate_image(image: Image.Image, findings: list) -> Image.Image:
    """วาดกรอบสีแดง/ส้มและป้ายกำกับภาษาไทยลงบนแบบแปลน"""
    draw_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    w, h = draw_img.size
    font_badge = load_thai_font(20)

    for idx, item in enumerate(findings, start=1):
        box = item.get("box_2d", [])
        if len(box) == 4:
            ymin, xmin, ymax, xmax = box
            left = (xmin / 1000) * w
            top = (ymin / 1000) * h
            right = (xmax / 1000) * w
            bottom = (ymax / 1000) * h

            is_error = item.get("type") == "error"
            color = "#DC2626" if is_error else "#EA580C"

            # วาดกรอบสี่เหลี่ยมรอบจุดผิด
            draw.rectangle([left, top, right, bottom], outline=color, width=5)
            
            # ป้ายข้อความกำกับภาษาไทย
            label_type = "ผิดพลาด" if is_error else "ข้อสังเกต"
            tag = f"#{idx} [{label_type}] {item.get('title', '')}"
            
            # วัดขนาดป้ายตามตัวอักษรจริง
            tag_w = draw.textlength(tag, font=font_badge) + 16
            tag_h = 30
            draw.rectangle([left, max(0, top - tag_h), left + tag_w, top], fill=color)
            draw.text((left + 8, max(0, top - tag_h + 2)), tag, font=font_badge, fill="white")

    return draw_img

def create_report_pdf_pages(filename: str, all_findings: list) -> list:
    """สร้างหน้ารายงานสรุปคอมเมนต์ภาษาไทย (กระดาษ A4) แนบต่อท้าย PDF"""
    A4_W, A4_H = 1240, 1754  # 150 DPI
    report_images = []
    
    font_header = load_thai_font(30)
    font_sub = load_thai_font(20)
    font_sheet = load_thai_font(24)
    font_badge = load_thai_font(20)
    font_desc = load_thai_font(20)

    current_img = Image.new("RGB", (A4_W, A4_H), color="white")
    draw = ImageDraw.Draw(current_img)
    y_cursor = 60

    # Header ส่วนหัวรายงาน
    draw.rectangle([50, 40, A4_W - 50, 150], fill="#1E3A8A")
    draw.text((80, 55), "รายงานผลการตรวจสอบแบบแปลนวิศวกรรมชลประทาน", font=font_header, fill="white")
    draw.text((80, 105), f"ชื่อไฟล์: {filename}   |   วันที่ตรวจสอบ: {datetime.now().strftime('%d/%m/%Y %H:%M')}", font=font_sub, fill="#BFDBFE")
    y_cursor = 180

    for page_idx, findings in enumerate(all_findings, start=1):
        if y_cursor > A4_H - 220:
            report_images.append(current_img)
            current_img = Image.new("RGB", (A4_W, A4_H), color="white")
            draw = ImageDraw.Draw(current_img)
            y_cursor = 60

        # แถบหัวข้อหน้า
        draw.rectangle([50, y_cursor, A4_W - 50, y_cursor + 45], fill="#F1F5F9")
        draw.text((70, y_cursor + 8), f"📄 แบบแปลนหน้าที่ {page_idx} (พบข้อสังเกต {len(findings)} รายการ)", font=font_sheet, fill="#0F172A")
        y_cursor += 60

        if not findings:
            draw.text((80, y_cursor), "🟢 ไม่พบข้อผิดพลาดหรือข้อสังเกตที่ชัดเจนในหน้านี้", font=font_desc, fill="#16A34A")
            y_cursor += 45
            continue

        for idx, item in enumerate(findings, start=1):
            if y_cursor > A4_H - 180:
                report_images.append(current_img)
                current_img = Image.new("RGB", (A4_W, A4_H), color="white")
                draw = ImageDraw.Draw(current_img)
                y_cursor = 60

            is_err = item.get("type") == "error"
            tag_color = "#DC2626" if is_err else "#EA580C"
            tag_title = f"#{idx} [{'จุดผิดพลาด' if is_err else 'ข้อสังเกต'}] {item.get('title', '')}"

            # หัวข้อรายการ
            draw.rectangle([70, y_cursor, 90, y_cursor + 24], fill=tag_color)
            draw.text((105, y_cursor - 2), tag_title, font=font_badge, fill=tag_color)
            y_cursor += 32

            # รายละเอียดคำอธิบาย ตัดคำภาษาไทยตามความกว้าง
            desc = f"รายละเอียด: {item.get('description', '')}"
            wrapped_lines = wrap_text_thai(desc, font_desc, max_width=A4_W - 200, draw=draw)
            for line in wrapped_lines:
                draw.text((105, y_cursor), line, font=font_desc, fill="#334155")
                y_cursor += 28
            y_cursor += 15

        y_cursor += 20

    report_images.append(current_img)
    return report_images

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
            findings, used_model = analyze_with_fallback(
                client=client,
                primary_model=selected_model,
                contents=[
                    types.Part.from_bytes(data=img_buffer.getvalue(), mime_type="image/jpeg"),
                    PROMPT
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
        except Exception as e:
            st.warning(f"⚠️ หน้าที่ {idx + 1} ไม่สามารถตรวจได้: {e}")
            findings = []

        annotated_img = annotate_image(page_img, findings)
        annotated_pages.append(annotated_img)
        all_findings.append(findings)

        progress_bar.progress((idx + 1) / len(raw_pages))
        time.sleep(2.0)

    status_text.empty()
    st.success("✅ ตรวจสอบแบบแปลนและจัดทำรายงานเสร็จสิ้นเรียบร้อยแล้ว!")

    # รวมหน้าแบบแปลน + หน้ารายงานภาษาไทย เข้าด้วยกันเป็นไฟล์ PDF เดียว
    report_pages = create_report_pdf_pages(uploaded_file.name, all_findings)
    final_combined_pdf_pages = annotated_pages + report_pages

    output_pdf = io.BytesIO()
    if final_combined_pdf_pages:
        final_combined_pdf_pages[0].save(
            output_pdf,
            format="PDF",
            save_all=True,
            append_images=final_combined_pdf_pages[1:]
        )
        output_pdf.seek(0)

        st.subheader("📥 ดาวน์โหลดเอกสารฉบับสมบูรณ์")
        st.download_button(
            label="📄 ดาวน์โหลดผลการตรวจสอบ (แบบแปลนวงจุดผิด + ใบรายงานภาษาไทยใน PDF เดียว)",
            data=output_pdf,
            file_name=f"Full_Audit_Report_{uploaded_file.name}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    st.divider()

    # แสดงพรีวิวหน้าเว็บ
    st.subheader("📋 รายละเอียดผลการตรวจสอบแต่ละหน้า")
    for i, (ann_img, page_finds) in enumerate(zip(annotated_pages, all_findings)):
        with st.expander(f"📄 แบบแปลนหน้าที่ {i + 1} (พบ {len(page_finds)} รายการ)", expanded=(i == 0)):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.image(ann_img, use_container_width=True, caption=f"ภาพแบบแปลนหน้าที่ {i + 1}")
            with col2:
                st.markdown("#### ข้อคิดเห็นทางวิศวกรรม:")
                if not page_finds:
                    st.write("🟢 ไม่พบข้อผิดพลาดที่ชัดเจนในหน้านี้")
                for f_idx, item in enumerate(page_finds, start=1):
                    badge = "🔴 ผิดพลาด (Error)" if item.get("type") == "error" else "🟠 ข้อสังเกต (Observation)"
                    st.markdown(f"**#{f_idx} [{badge}] {item.get('title')}**")
                    st.write(item.get("description"))
                    st.divider()
