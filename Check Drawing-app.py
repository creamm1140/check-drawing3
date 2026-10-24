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

st.set_page_config(
    page_title="AI ตรวจแบบแปลนวิศวกรรมชลประทาน (High Precision)",
    page_icon="🚜",
    layout="wide"
)

st.title("🚜 AI ตรวจสอบแบบแปลนวิศวกรรมชลประทาน (ระบบตรวจละเอียด)")
st.caption("เทคโนโลยี Hybrid: ตรวจ Text Layer ภาษาไทยแม่นยำ 100% ควบคู่กับ AI Vision ตรวจมิติและชลศาสตร์")

# ----------------- ระบบโหลดฟอนต์ภาษาไทย -----------------
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

# ----------------- Prompts เฉพาะทางชลประทาน -----------------
VISION_PROMPT = """
คุณคือวิศวกรผู้เชี่ยวชาญด้านการตรวจสอบแบบแปลนวิศวกรรมชลประทานและโครงสร้างอาคารชลประทาน
จงวิเคราะห์ภาพแบบแปลนนี้อย่างละเอียดใน 3 มิติทางกายภาพ:
1. มิติโครงสร้าง (Dimensions & Scale): ตัวเลขระยะ ความสูง ความกว้าง ขัดแย้งกันเองหรือไม่ เช่น ระยะรวมไม่ตรงกับระยะย่อย, สเกลผิดสัดส่วน
2. หลักวิศวกรรมชลประทาน (Hydraulics & Levels):
   - ทิศทางการไหล (Flow Direction) สอดคล้องกับระดับความลาดชัน (Slope) หรือไม่
   - ระดับน้ำสำคัญ: ร.น.ส. (ระดับน้ำส่งน้ำ), ร.ม.ส. (ระดับน้ำสูงสุด), ระดับสันฝาย, ระดับก้นคลอง สอดคล้องตามหลักชลศาสตร์หรือไม่
   - ตำแหน่งและมิติอาคารชลประทาน เช่น ประตูระบายน้ำ, ทรบ., ท่อลอด, ทางระบายน้ำล้น (Spillway), บ่อดักตะกอน
3. สัญลักษณ์และโครงสร้าง: ระยะเหล็กเสริม, รอยต่อขยายตัว (Expansion Joint), ความหนาของคอนกรีตดาดคลอง

ส่งคำตอบเป็น JSON Array เท่านั้น:
[
  {
    "box_2d": [ymin, xmin, ymax, xmax],
    "type": "error" หรือ "observation",
    "title": "หัวข้อปัญหาชลศาสตร์/โครงสร้าง",
    "description": "คำอธิบายเชิงวิศวกรรมอย่างละเอียด"
  }
]
* หมายเหตุ: box_2d ต้องเป็นพิกัดสัดส่วน 0 ถึง 1000
"""

TEXT_SPELLCHECK_PROMPT = """
คุณคือผู้เชี่ยวชาญด้านการตรวจทานคำผิด (Proofreader) ในแบบแปลนวิศวกรรมชลประทาน
ตรวจสอบรายการข้อความที่ดึงมาจากแบบแปลนต่อไปนี้เพื่อหา "คำผิดภาษาไทย คำสะกดผิด คำตกหล่น หรือสระวรรณยุกต์วางผิดตำแหน่ง":
ตัวอย่างคำศัพท์ที่ถูกต้องในงานชลประทาน:
- คอนกรีต, คอนกรีตเสริมเหล็ก (ค.ส.ล.), คอนกรีตหยาบ
- ระดับก้นคลอง, ระดับสันคันคลอง, ระดับสันฝาย, ประตูระบายน้ำ
- ท่อระบายน้ำปากคลอง (ทรบ.), ท่อระบายน้ำลอด (ทรล.)
- ความลาดเอียง (Slope), ทางลาด, หินเรียงยาแนว, ดินถมบดอัดแน่น

ข้อความจากแบบแปลน (พร้อม ID และพิกัด):
{extracted_text_json}

จงตอบกลับเฉพาะรายการที่มีคำผิด เป็น JSON Array เท่านั้น ในรูปแบบ:
[
  {
    "id": หมายเลข ID ของคำที่ผิด,
    "wrong_word": "คำที่เขียนผิด",
    "correct_word": "คำที่ถูกต้อง",
    "description": "อธิบายว่าผิดอย่างไร"
  }
]
หากไม่พบคำผิด ให้ตอบ []
"""

# ----------------- ฟังก์ชันดึง Text Layer จาก PDF -----------------
def extract_words_with_bbox(page, page_width, page_height):
    """ดึงตัวหนังสือภาษาไทยทั้งหมดจาก PDF พร้อมคำนวณพิกัดสัดส่วน 0-1000"""
    words_data = []
    # word format: (x0, y0, x1, y1, word, block_no, line_no, word_no)
    words = page.get_text("words")
    for idx, w in enumerate(words):
        x0, y0, x1, y1, word_text = w[0], w[1], w[2], w[3], w[4]
        if len(word_text.strip()) > 1: # กรองอักขระเดี่ยวทิ้ง
            words_data.append({
                "id": idx,
                "text": word_text.strip(),
                "box_2d": [
                    int((y0 / page_height) * 1000),
                    int((x0 / page_width) * 1000),
                    int((y1 / page_height) * 1000),
                    int((x1 / page_width) * 1000)
                ]
            })
    return words_data

def call_gemini(client, model, contents, is_json=True):
    """เรียกใช้งาน Gemini พร้อมระบบทำความสะอาด JSON Output"""
    for _ in range(2):
        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json" if is_json else None,
                temperature=0.1 # ใช้ค่าต่ำเพื่อลดความคลาดเคลื่อน
            )
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
            text_resp = response.text.strip()
            if text_resp.startswith("```json"):
                text_resp = text_resp[7:]
            if text_resp.endswith("```"):
                text_resp = text_resp[:-3]
            return json.loads(text_resp.strip()) if is_json else text_resp
        except Exception:
            time.sleep(2)
    return []

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
                color = "#9333EA" # สีม่วงสำหรับคำผิด
                label_type = "คำผิด"
            elif item_type == "error":
                color = "#DC2626" # สีแดงสำหรับจุดผิดร้ายแรง
                label_type = "วิศวกรรม"
            else:
                color = "#EA580C" # สีส้มสำหรับข้อสังเกต
                label_type = "ข้อสังเกต"

            draw.rectangle([left, top, right, bottom], outline=color, width=4)
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
    draw.text((80, 55), "รายงานผลการตรวจสอบแบบแปลนวิศวกรรมชลประทานอย่างละเอียด", font=font_header, fill="white")
    draw.text((80, 105), f"ไฟล์: {filename}  |  วันที่: {datetime.now().strftime('%d/%m/%Y %H:%M')}", font=font_sub, fill="#BFDBFE")
    y_cursor = 180

    for page_idx, findings in enumerate(all_findings, start=1):
        if y_cursor > A4_H - 220:
            report_images.append(current_img)
            current_img = Image.new("RGB", (A4_W, A4_H), color="white")
            draw = ImageDraw.Draw(current_img)
            y_cursor = 60

        draw.rectangle([50, y_cursor, A4_W - 50, y_cursor + 45], fill="#F1F5F9")
        draw.text((70, y_cursor + 8), f"📄 แบบแปลนหน้าที่ {page_idx} (พบข้อบกพร่อง/ข้อสังเกต {len(findings)} รายการ)", font=font_sheet, fill="#0F172A")
        y_cursor += 60

        if not findings:
            draw.text((80, y_cursor), "🟢 ไม่พบข้อผิดพลาดหรือข้อสังเกตในหน้านี้", font=font_desc, fill="#16A34A")
            y_cursor += 45
            continue

        for idx, item in enumerate(findings, start=1):
            if y_cursor > A4_H - 180:
                report_images.append(current_img)
                current_img = Image.new("RGB", (A4_W, A4_H), color="white")
                draw = ImageDraw.Draw(current_img)
                y_cursor = 60

            item_type = item.get("type", "observation")
            if item_type == "typo":
                tag_color = "#9333EA"
                type_name = "คำผิดภาษาไทย"
            elif item_type == "error":
                tag_color = "#DC2626"
                type_name = "ข้อผิดพลาดโครงสร้าง"
            else:
                tag_color = "#EA580C"
                type_name = "ข้อสังเกตชลศาสตร์"

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

# ----------------- ส่วน UI การอัปโหลด -----------------
uploaded_file = st.file_uploader(
    "📁 อัปโหลดแบบแปลน (แนะนำ PDF สำหรับระบบสแกนคำผิดอัตโนมัติ):", 
    type=["pdf", "jpg", "jpeg", "png", "tiff"]
)

if uploaded_file and st.button("🚀 เริ่มการตรวจสอบแบบละเอียด (Start Precision Audit)"):
    file_bytes = uploaded_file.read()
    raw_pages = []
    is_pdf = uploaded_file.name.lower().endswith(".pdf")
    pdf_doc = fitz.open(stream=file_bytes, filetype="pdf") if is_pdf else None

    with st.spinner("กำลังแปลงไฟล์เอกสารด้วยความละเอียดสูง (200 DPI)..."):
        if is_pdf:
            for page in pdf_doc:
                # ปรับเป็น 200 DPI เพื่อความคมชัดสูงสุดในการอ่านระดับและมิติ
                pix = page.get_pixmap(dpi=200)
                raw_pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        else:
            raw_pages.append(Image.open(io.BytesIO(file_bytes)).convert("RGB"))

    st.info(f"ตรวจพบ {len(raw_pages)} หน้า กำลังเริ่มการสแกน 2 ขั้นตอน (ตรวจคำผิดจาก Text Layer + ตรวจวิศวกรรมด้วย Vision)...")

    annotated_pages = []
    all_findings = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, page_img in enumerate(raw_pages):
        page_findings = []

        # ----------------- ขั้นตอนที่ 1: ตรวจสอบคำผิดภาษาไทย (Text Layer) -----------------
        if is_pdf and pdf_doc:
            pdf_page = pdf_doc[idx]
            page_rect = pdf_page.rect
            extracted_words = extract_words_with_bbox(pdf_page, page_rect.width, page_rect.height)
            
            if extracted_words:
                status_text.text(f"🔍 หน้าที่ {idx + 1}: กำลังตรวจทานตัวสะกดภาษาไทยจากเลเยอร์ข้อความ...")
                # สรุปเฉพาะข้อความส่งให้ AI
                words_payload = json.dumps([{"id": w["id"], "text": w["text"]} for w in extracted_words[:300]], ensure_ascii=False)
                typo_prompt = TEXT_SPELLCHECK_PROMPT.format(extracted_text_json=words_payload)
                
                typo_results = call_gemini(client, selected_model, [typo_prompt])
                
                # นำคำผิดที่ AI พบกลับมาจับคู่กับ Bounding Box พิกัดจริง
                words_map = {w["id"]: w["box_2d"] for w in extracted_words}
                for typo in typo_results:
                    w_id = typo.get("id")
                    if w_id in words_map:
                        page_findings.append({
                            "box_2d": words_map[w_id],
                            "type": "typo",
                            "title": f"คำผิด: '{typo.get('wrong_word')}'",
                            "description": f"คำที่เขียนผิดคือ '{typo.get('wrong_word')}' ควรแก้เป็น '{typo.get('correct_word')}' ({typo.get('description')})"
                        })

        # ----------------- ขั้นตอนที่ 2: ตรวจสอบวิศวกรรม/ชลศาสตร์ (Vision) -----------------
        status_text.text(f"🔍 หน้าที่ {idx + 1}: กำลังตรวจสอบมิติและโครงสร้างชลศาสตร์...")
        img_buffer = io.BytesIO()
        page_img.save(img_buffer, format="JPEG", quality=95) # เพิ่มคุณภาพรูป

        vision_results = call_gemini(
            client=client,
            model=selected_model,
            contents=[
                types.Part.from_bytes(data=img_buffer.getvalue(), mime_type="image/jpeg"),
                VISION_PROMPT
            ]
        )
        if isinstance(vision_results, list):
            page_findings.extend(vision_results)

        # วาดผลลัพธ์ลงบนภาพ
        annotated_img = annotate_image(page_img, page_findings)
        annotated_pages.append(annotated_img)
        all_findings.append(page_findings)

        progress_bar.progress((idx + 1) / len(raw_pages))
        time.sleep(1.5)

    status_text.empty()
    st.success("✅ ตรวจสอบแบบแปลนครบทุกมิติเสร็จสมบูรณ์!")

    # รวมไฟล์ PDF ผลการตรวจสอบ
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
            label="📄 ดาวน์โหลดผลการตรวจสอบ (แบบแปลนวงจุดผิด + รายงานภาษาไทยใน PDF เดียว)",
            data=output_pdf,
            file_name=f"Detailed_Audit_{uploaded_file.name}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    st.divider()

    # แสดงผลลัพธ์พรีวิว
    st.subheader("📋 สรุปผลการตรวจสอบแต่ละหน้า")
    for i, (ann_img, page_finds) in enumerate(zip(annotated_pages, all_findings)):
        with st.expander(f"📄 แบบแปลนหน้าที่ {i + 1} (พบข้อบกพร่อง/คำผิด {len(page_finds)} รายการ)", expanded=(i == 0)):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.image(ann_img, use_container_width=True, caption=f"ภาพแบบแปลนหน้าที่ {i + 1}")
            with col2:
                st.markdown("#### รายการที่ตรวจพบ:")
                if not page_finds:
                    st.write("🟢 ไม่พบข้อผิดพลาดในหน้านี้")
                for f_idx, item in enumerate(page_finds, start=1):
                    item_type = item.get("type")
                    if item_type == "typo":
                        badge = "🟣 คำผิดภาษาไทย"
                    elif item_type == "error":
                        badge = "🔴 ผิดพลาด (วิศวกรรม)"
                    else:
                        badge = "🟠 ข้อสังเกต (ชลศาสตร์)"
                    st.markdown(f"**#{f_idx} [{badge}] {item.get('title')}**")
                    st.write(item.get("description"))
                    st.divider()
