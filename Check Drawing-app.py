import streamlit as st
import io
import json
import time
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
st.caption("ระบบตรวจสอบมิติโครงสร้าง, ชลศาสตร์, สัญลักษณ์ และคำผิดอัตโนมัติ (รวมรายงานใน PDF เดียว)")

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

# ----------------- ฟังก์ชันประมวลผล & Fallback -----------------
def analyze_with_fallback(client, primary_model, contents, config, fallback_model="gemini-3.5-flash-lite"):
    """ส่งตรวจพร้อมระบบ Auto-Retry และ Fallback สลับโมเดลอัตโนมัติหากติด 429 หรือ 503"""
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
    """วาดกรอบสีแดง/ส้มและป้ายหมายเลขกำกับลงบนภาพแบบแปลน"""
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

            # วาดกรอบสี่เหลี่ยมรอบจุดผิด
            draw.rectangle([left, top, right, bottom], outline=color, width=5)
            
            # ป้ายแท็กหมายเลขกำกับ เช่น #1 [ERROR]
            tag = f"#{idx} {'ERR' if is_error else 'OBS'}: {item.get('title', '')}"
            draw.rectangle([left, max(0, top - 25), left + max(len(tag) * 9, 80), top], fill=color)
            draw.text((left + 5, max(0, top - 20)), tag, fill="white")

    return draw_img

def create_report_pdf_pages(filename: str, all_findings: list) -> list:
    """สร้างหน้ารายงานสรุปคอมเมนต์ลงกระดาษ A4 เพื่อนำไปแนบต่อท้าย PDF"""
    # ขนาดกระดาษ A4 แนวนอน/แนวตั้ง (ใช้แนวตั้ง 1240 x 1754 px ที่ 150 DPI)
    A4_W, A4_H = 1240, 1754
    report_images = []
    
    current_img = Image.new("RGB", (A4_W, A4_H), color="white")
    draw = ImageDraw.Draw(current_img)
    y_cursor = 60

    # Header หัวรายงาน
    draw.rectangle([50, 40, A4_W - 50, 140], fill="#1E3A8A")
    draw.text((80, 60), "IRRIGATION ENGINEERING DRAWING AUDIT REPORT", fill="white")
    draw.text((80, 95), f"File: {filename}  |  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fill="#93C5FD")
    y_cursor = 170

    for page_idx, findings in enumerate(all_findings, start=1):
        # ตรวจสอบพื้นที่ ถ้าเนื้อหาใกล้ล้นหน้ากระดาษ ให้ตัดขึ้นหน้า A4 แผ่นใหม่
        if y_cursor > A4_H - 220:
            report_images.append(current_img)
            current_img = Image.new("RGB", (A4_W, A4_H), color="white")
            draw = ImageDraw.Draw(current_img)
            y_cursor = 60

        # แถบหัวข้อหน้าแบบแปลน
        draw.rectangle([50, y_cursor, A4_W - 50, y_cursor + 35], fill="#E2E8F0")
        draw.text((65, y_cursor + 8), f"SHEET / PAGE {page_idx} (Found {len(findings)} Items)", fill="#0F172A")
        y_cursor += 45

        if not findings:
            draw.text((80, y_cursor), "- No significant engineering errors detected on this page.", fill="#16A34A")
            y_cursor += 35
            continue

        for idx, item in enumerate(findings, start=1):
            if y_cursor > A4_H - 180:
                report_images.append(current_img)
                current_img = Image.new("RGB", (A4_W, A4_H), color="white")
                draw = ImageDraw.Draw(current_img)
                y_cursor = 60

            is_err = item.get("type") == "error"
            tag_color = "#DC2626" if is_err else "#EA580C"
            tag_text = f"#{idx} [ERROR]" if is_err else f"#{idx} [OBSERVATION]"
            
            # กล่องป้ายประเภท
            draw.rectangle([70, y_cursor, 220, y_cursor + 24], fill=tag_color)
            draw.text((78, y_cursor + 4), tag_text, fill="white")
            
            # หัวข้อข้อสังเกต
            title_text = item.get("title", "")
            draw.text((235, y_cursor + 4), title_text, fill="#0F172A")
            y_cursor += 30

            # รายละเอียดคำอธิบาย (ตัดบรรทัดเพื่อไม่ให้ล้นขอบขวา)
            desc = item.get("description", "")
            # แบ่งบรรทัดทีละประมาณ 90 ตัวอักษร
            words = desc.split(" ")
            line_buf = "   "
            for w in words:
                if len(line_buf) + len(w) > 85:
                    draw.text((70, y_cursor), line_buf, fill="#334155")
                    y_cursor += 22
                    line_buf = "   " + w + " "
                else:
                    line_buf += w + " "
            if line_buf.strip():
                draw.text((70, y_cursor), line_buf, fill="#334155")
                y_cursor += 28

        y_cursor += 15

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

    # ----------------- รวมหน้ารายงานคอมเมนต์ต่อท้ายใน PDF -----------------
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

        # ----------------- ปุ่มดาวน์โหลด PDF ครบจบในไฟล์เดียว -----------------
        st.subheader("📥 ดาวน์โหลดเอกสารผลการตรวจสอบฉบับสมบูรณ์")
        st.download_button(
            label="📄 ดาวน์โหลดผลการตรวจสอบฉบับรวม (แบบแปลนวงจุดผิด + ใบรายงานคอมเมนต์ในไฟล์เดียว)",
            data=output_pdf,
            file_name=f"Full_Audit_Report_{uploaded_file.name}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    st.divider()

    # ----------------- แสดงตัวอย่างการตรวจสอบบนหน้าเว็บ -----------------
    st.subheader("📋 ตัวอย่างผลการตรวจสอบแต่ละหน้า")
    for i, (ann_img, page_finds) in enumerate(zip(annotated_pages, all_findings)):
        with st.expander(f"📄 แบบแปลนหน้าที่ {i + 1} (พบ {len(page_finds)} รายการ)", expanded=(i == 0)):
            col1, col2 = st.columns([3, 2])
            with col1:
                st.image(ann_img, use_container_width=True, caption=f"ภาพแบบแปลนหน้าที่ {i + 1}")
            with col2:
                st.markdown("#### รายการข้อคิดเห็น:")
                if not page_finds:
                    st.write("🟢 ไม่พบข้อผิดพลาดที่ชัดเจนในหน้านี้")
                for f_idx, item in enumerate(page_finds, start=1):
                    badge = "🔴 ผิดพลาด (Error)" if item.get("type") == "error" else "🟠 ข้อสังเกต (Observation)"
                    st.markdown(f"**#{f_idx} [{badge}] {item.get('title')}**")
                    st.write(item.get("description"))
                    st.divider()
