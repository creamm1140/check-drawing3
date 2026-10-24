import streamlit as st
import io
import json
import time
import os
import urllib.request
from datetime import datetime
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field
from typing import List, Literal
from google import genai
from google.genai import types

# ----------------- ตั้งค่าหน้าเว็บ -----------------
st.set_page_config(
    page_title="AI ตรวจแบบแปลนวิศวกรรมชลประทาน (Deterministic Engine)",
    page_icon="🚜",
    layout="wide"
)

st.title("🚜 AI ตรวจสอบแบบแปลนวิศวกรรมชลประทาน (ระบบความเสถียรสูง)")
st.caption("Single-Pass Multimodal + Structured Outputs ล็อกผลลัพธ์ให้นิ่ง ไม่หลุดโควตา")

# ----------------- กำหนด Schema บังคับโครงสร้าง JSON -----------------
class FindingItem(BaseModel):
    box_2d: List[int] = Field(description="พิกัด [ymin, xmin, ymax, xmax] สัดส่วน 0 ถึง 1000")
    type: Literal["error", "observation", "typo"] = Field(description="ประเภท: error (วิศวกรรม), observation (ข้อสังเกต), typo (คำผิด)")
    title: str = Field(description="หัวข้อจุดบกพร่องสั้นๆ กระชับ")
    description: str = Field(description="คำอธิบายเชิงวิศวกรรมอย่างละเอียด")

class PageAuditResult(BaseModel):
    findings: List[FindingItem] = Field(default_factory=list)

# ----------------- ระบบฟอนต์ภาษาไทย -----------------
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
FONT_PATH = "Sarabun-Regular.ttf"

@st.cache_resource
def load_thai_font(size: int = 20):
    if not os.path.exists(FONT_PATH):
        try:
            urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        except Exception:
            pass
    if os.path.exists(FONT_PATH):
        try:
            return ImageFont.truetype(FONT_PATH, size)
        except Exception:
            pass
    for win_font in ["C:/Windows/Fonts/tahoma.ttf", "C:/Windows/Fonts/angsana.ttf"]:
        if os.path.exists(win_font):
            try:
                return ImageFont.truetype(win_font, size)
            except Exception:
                pass
    return ImageFont.load_default()

def wrap_text_thai(text: str, font, max_width: int, draw: ImageDraw.Draw) -> list:
    lines = []
    current_line = ""
    for char in text:
        test_line = current_line + char
        if draw.textlength(test_line, font=font) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = char
    if current_line:
        lines.append(current_line)
    return lines

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("⚙️ การตั้งค่าระบบ")
    default_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = st.text_input("Gemini API Key:", value=default_key, type="password")
    selected_model = st.selectbox(
        "เลือกรุ่นโมเดล AI:",
        ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        index=0
    )
    st.markdown("[👉 รับ API Key ฟรีที่ Google AI Studio](https://aistudio.google.com/)")

if not api_key:
    st.warning("⚠️ กรุณากรอก Gemini API Key ในแถบด้านซ้ายก่อนเริ่มใช้งาน")
    st.stop()

try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อ Gemini API: {e}")
    st.stop()

# ----------------- Prompt ตรวจสอบรวม (Single-Pass) -----------------
MASTER_PROMPT = """
คุณคือหัวหน้าวิศวกรผู้เชี่ยวชาญด้านการตรวจสอบแบบแปลนวิศวกรรมชลประทาน (กรมชลประทาน)
จงตรวจสอบแบบแปลนในภาพนี้ พร้อมข้อความ Text Layer ที่แนบมาอย่างละเอียดที่สุดในทุกมิติ:

1. ข้อผิดพลาดทางวิศวกรรมโครงสร้างและมิติ (type: "error"):
   - ข้อมูลขัดแย้งระหว่าง Title Block กับรายการประกอบแบบ (เช่น หมู่บ้าน, ตำบล, อำเภอ)
   - ตัวเลขมิติรวมไม่ตรงกับมิติย่อย, ความกว้างช่องระบายน้ำขัดแย้งกับขนาดตอม่อ/เสากลาง
   - ระดับน้ำขัดแย้งกับการไหล (Gravity Flow) เช่น ท้ายน้ำสูงกว่าเหนือน้ำ
   - ข้อกำหนดเหล็กเสริม, ระยะทาบเหล็ก, Cutoff Trench ไม่ครบถ้วน

2. ข้อสังเกตและมาตรฐานการเขียนแบบ (type: "observation"):
   - ตัวเลข/ตัวอักษรกลับทิศทาง (Inverted/Upside-down)
   - สัญลักษณ์ชลศาสตร์, ทิศเหนือ, มาตราส่วน (Scale)
   - ความสมมาตรของอาคารอัดน้ำเทียบกับแนวศูนย์กลาง (CL)

3. คำผิด/คำตกหล่นภาษาไทย (type: "typo"):
   - ตรวจสอบคำสะกดผิดในข้อความภาษาไทย เช่น สันฝาย, คอนกรีต, คลองดาด, ทรบ., หินเรียง

ข้อความ Text Layer ที่สกัดได้จากแบบแปลนหน้านี้:
{extracted_text}
"""

def extract_text_from_page(pdf_page):
    """สกัดข้อความทั้งหมดในหน้านั้นเพื่อส่งคู่กับรูปภาพ"""
    text = pdf_page.get_text("text")
    return text[:4000] if text else "ไม่มี Text Layer ดิจิทัล (เป็นภาพสแกน)"

def annotate_image(image: Image.Image, findings: list) -> Image.Image:
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

            item_type = item.get("type", "observation")
            if item_type == "typo":
                color = "#9333EA"
                label_type = "คำผิด"
            elif item_type == "error":
                color = "#DC2626"
                label_type = "ผิดพลาด"
            else:
                color = "#EA580C"
                label_type = "ข้อสังเกต"

            draw.rectangle([left, top, right, bottom], outline=color, width=5)
            tag = f"#{idx} [{label_type}] {item.get('title', '')}"
            tag_w = draw.textlength(tag, font=font_badge) + 16
            draw.rectangle([left, max(0, top - 28), left + tag_w, top], fill=color)
            draw.text((left + 8, max(0, top - 26)), tag, font=font_badge, fill="white")

    return draw_img

def create_report_pdf_pages(filename: str, all_findings: list) -> list:
    A4_W, A4_H = 1240, 1754
    report_images = []
    
    font_header = load_thai_font(30)
    font_sub = load_thai_font(20)
    font_sheet = load_thai_font(24)
    font_badge = load_thai_font(20)
    font_desc = load_thai_font(18)

    current_img = Image.new("RGB", (A4_W, A4_H), color="white")
    draw = ImageDraw.Draw(current_img)
    y_cursor = 60

    draw.rectangle([50, 40, A4_W - 50, 150], fill="#1E3A8A")
    draw.text((80, 55), "รายงานผลการตรวจสอบแบบแปลนวิศวกรรมชลประทาน", font=font_header, fill="white")
    draw.text((80, 105), f"ไฟล์: {filename}  |  วันที่: {datetime.now().strftime('%d/%m/%Y %H:%M')}", font=font_sub, fill="#BFDBFE")
    y_cursor = 180

    for page_idx, findings in enumerate(all_findings, start=1):
        if y_cursor > A4_H - 220:
            report_images.append(current_img)
            current_img = Image.new("RGB", (A4_W, A4_H), color="white")
            draw = ImageDraw.Draw(current_img)
            y_cursor = 60

        draw.rectangle([50, y_cursor, A4_W - 50, y_cursor + 45], fill="#F1F5F9")
        draw.text((70, y_cursor + 8), f"📄 แบบแปลนหน้าที่ {page_idx} (พบ {len(findings)} รายการ)", font=font_sheet, fill="#0F172A")
        y_cursor += 60

        if not findings:
            draw.text((80, y_cursor), "🟢 ไม่พบข้อบกพร่องหรือข้อสังเกตในหน้านี้", font=font_desc, fill="#16A34A")
            y_cursor += 45
            continue

        for idx, item in enumerate(findings, start=1):
            if y_cursor > A4_H - 180:
                report_images.append(current_img)
                current_img = Image.new("RGB", (A4_W, A4_H), color="white")
                draw = ImageDraw.Draw(current_img)
                y_cursor = 60

            item_type = item.get("type", "observation")
            tag_color = "#9333EA" if item_type == "typo" else ("#DC2626" if item_type == "error" else "#EA580C")
            type_name = "คำผิดภาษาไทย" if item_type == "typo" else ("ข้อผิดพลาดโครงสร้าง" if item_type == "error" else "ข้อสังเกตชลศาสตร์")

            tag_title = f"#{idx} [{type_name}] {item.get('title', '')}"
            draw.rectangle([70, y_cursor, 90, y_cursor + 24], fill=tag_color)
            draw.text((105, y_cursor - 2), tag_title, font=font_badge, fill=tag_color)
            y_cursor += 30

            desc = f"รายละเอียด: {item.get('description', '')}"
            wrapped_lines = wrap_text_thai(desc, font_desc, max_width=A4_W - 200, draw=draw)
            for line in wrapped_lines:
                draw.text((105, y_cursor), line, font=font_desc, fill="#334155")
                y_cursor += 26
            y_cursor += 12

        y_cursor += 20

    report_images.append(current_img)
    return report_images

# ----------------- UI และระบบประมวลผล -----------------
uploaded_file = st.file_uploader("📁 อัปโหลดแบบแปลนชลประทาน (PDF, JPG, PNG):", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file and st.button("🚀 เริ่มการตรวจสอบ (Deterministic Audit)"):
    file_bytes = uploaded_file.read()
    raw_pages = []
    page_texts = []
    is_pdf = uploaded_file.name.lower().endswith(".pdf")
    
    with st.spinner("กำลังแปลงไฟล์เอกสาร..."):
        if is_pdf:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=180) # คมชัดและไม่กิน Payload
                raw_pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
                page_texts.append(extract_text_from_page(page))
        else:
            raw_pages.append(Image.open(io.BytesIO(file_bytes)).convert("RGB"))
            page_texts.append("ไม่มี Text Layer (ไฟล์รูปภาพ)")

    st.info(f"ตรวจพบทั้งหมด {len(raw_pages)} หน้า เริ่มการตรวจสอบแบบ Single-Pass...")

    annotated_pages = []
    all_findings = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, (page_img, p_text) in enumerate(zip(raw_pages, page_texts)):
        status_text.text(f"⏳ กำลังตรวจสอบหน้าที่ {idx + 1}/{len(raw_pages)} อย่างละเอียด...")
        
        img_buffer = io.BytesIO()
        page_img.save(img_buffer, format="JPEG", quality=90)
        
        page_findings = []
        success = False
        error_msg = ""

        # Auto-retry พร้อมล็อกอุณหภูมิเป็น 0.0 ( Deterministic )
        for attempt in range(1, 4):
            try:
                prompt_content = MASTER_PROMPT.format(extracted_text=p_text)
                response = client.models.generate_content(
                    model=selected_model,
                    contents=[
                        types.Part.from_bytes(data=img_buffer.getvalue(), mime_type="image/jpeg"),
                        prompt_content
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0, # บังคับผลลัพธ์ให้นิ่ง ไม่สุ่มเดา
                        response_mime_type="application/json",
                        response_schema=PageAuditResult # บังคับโครงสร้าง JSON 100%
                    )
                )
                
                result_obj = PageAuditResult.model_validate_json(response.text)
                page_findings = [f.model_dump() for f in result_obj.findings]
                success = True
                break

            except Exception as e:
                error_msg = str(e)
                # หากเจอ 503 หรือ 429 ให้รอแล้วสู้ต่อ
                if any(k in error_msg for k in ["503", "429", "RESOURCE_EXHAUSTED", "UNAVAILABLE"]):
                    time.sleep(attempt * 4) # หน่วงเวลา 4s, 8s, 12s
                    continue
                else:
                    break

        if not success:
            st.error(f"❌ หน้าที่ {idx + 1} ไม่สามารถตรวจสอบได้: {error_msg}")
        
        annotated_img = annotate_image(page_img, page_findings)
        annotated_pages.append(annotated_img)
        all_findings.append(page_findings)

        progress_bar.progress((idx + 1) / len(raw_pages))
        time.sleep(3.0) # เว้นจังหวะ 3 วินาที เพื่อไม่ให้ชน Rate limit ของ Google

    status_text.empty()
    st.success("✅ การตรวจสอบเสร็จสมบูรณ์!")

    # รวมไฟล์ PDF
    report_pages = create_report_pdf_pages(uploaded_file.name, all_findings)
    final_combined = annotated_pages + report_pages

    output_pdf = io.BytesIO()
    if final_combined:
        final_combined[0].save(
            output_pdf,
            format="PDF",
            save_all=True,
            append_images=final_combined[1:]
        )
        output_pdf.seek(0)

        st.subheader("📥 ดาวน์โหลดรายงานฉบับสมบูรณ์")
        st.download_button(
            label="📄 ดาวน์โหลด Full Audit PDF (แปลนวงจุดผิด + ใบสรุปผลภาษาไทย)",
            data=output_pdf,
            file_name=f"Verified_Audit_{uploaded_file.name}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    st.divider()

    # แสดงผลบนหน้าเว็บ
    for i, (ann_img, finds) in enumerate(zip(annotated_pages, all_findings)):
        with st.expander(f"📄 แบบแปลนหน้าที่ {i + 1} (พบ {len(finds)} รายการ)", expanded=(i == 0)):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.image(ann_img, use_container_width=True)
            with col2:
                if not finds:
                    st.write("🟢 ไม่พบข้อบกพร่องในหน้านี้")
                for f_idx, item in enumerate(finds, start=1):
                    t = item.get("type")
                    badge = "🟣 คำผิด" if t == "typo" else ("🔴 ผิดพลาด" if t == "error" else "🟠 ข้อสังเกต")
                    st.markdown(f"**#{f_idx} [{badge}] {item.get('title')}**")
                    st.write(item.get("description"))
                    st.divider()
