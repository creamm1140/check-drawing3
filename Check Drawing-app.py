import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io
import json
import gc
import fitz  # PyMuPDF
from PIL import Image
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Hydraulic & Civil Review Pro", layout="wide")
st.title("ระบบผู้ช่วยตรวจแบบ: วิศวกรรมชลประทาน วิศวกรรมโยธา (Pro)")

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-3.5-flash-lite') 
except Exception as e:
    model = None
    st.error(f"❌ เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}")

# 1. System Prompt
system_prompt = """
คุณคือวิศวกรชลประทาน วิศวกรโยธา และผู้พิสูจน์อักษรระดับเชี่ยวชาญ
ตรวจสอบรูปภาพแบบแปลนหรือรายการคำนวณที่แนบมานี้อย่างละเอียดที่สุด:

1. วิศวกรรม (Engineering): ตรวจสอบความถูกต้องทางด้านวิศวกรรมชลประทาน และวิศวกรรมโยธา เช่น แรงดันน้ำ (Uplift), Factor of Safety, ความหนาคอนกรีต, ระดับ Elevation, มิติอาคาร, ความกว้าง ยาว สูง 
2. คำผิด (Typo): สแกนหาคำผิดทุกตัวอักษร ทั้งภาษาไทยและศัพท์เทคนิคภาษาอังกฤษ
3. ข้อสังเกต (Observation): หากไม่พบจุดผิดเลย ให้เสนอแนะจุดที่ควรระวังในการก่อสร้าง

คุณต้องตอบกลับมาเป็น JSON เท่านั้น โครงสร้างดังนี้:
{
  "project_type": "ประเภทของโครงการ/งานที่ตรวจพบในหน้านี้",
  "issues": [
    {
      "type": "Typo หรือ Engineering หรือ Observation",
      "error_keyword": "เลือกคำที่ผิดเพียง 1-2 คำที่ปรากฏในแบบเป๊ะๆ เพื่อให้ระบบค้นหาพิกัดเจอ (ห้ามพิมพ์เป็นประโยคยาว)",
      "description": "อธิบายว่าผิดอย่างไร และควรแก้ไขเป็นอะไร",
      "box_2d": [ymin, xmin, ymax, xmax]
    }
  ],
  "summary": "สรุปผลการตรวจสอบของหน้านี้"
}

กฎพิกัด box_2d:
- ใช้สเกล 0 ถึง 1000 สำหรับรูปภาพในหน้านั้นๆ (ซ้ายบน 0,0 ขวาล่าง 1000,1000)
- หากเป็นคำแนะนำภาพรวมที่ไม่มีจุดบนแบบ ให้ส่ง [0, 0, 0, 0]
"""

def clean_json_response(text):
    try:
        cleaned = text.replace('```json', '').replace('```', '').strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

def draw_and_annotate_on_pdf(page, issues):
    """ระบบวาดวงกลมแบบ Hybrid (ค้นหาข้อความ + ใช้พิกัด AI สำรอง)"""
    rect_width = page.rect.width
    rect_height = page.rect.height
    
    for issue in issues:
        keyword = issue.get("error_keyword", "").strip()
        desc = issue.get("description", "ไม่ระบุรายละเอียด")
        i_type = issue.get("type", "Observation")
        box = issue.get("box_2d", [0, 0, 0, 0])
        
        annot_text = f"[{i_type.upper()}]\n{desc}"
        found_coordinates = False
        
        # วิธีที่ 1: ค้นหาข้อความตรงๆ ใน PDF
        if keyword:
            text_instances = page.search_for(keyword, quads=True) 
            if text_instances:
                found_coordinates = True
                for quad in text_instances:
                    rect = quad.rect
                    rect_expanded = fitz.Rect(rect.x0 - 5, rect.y0 - 5, rect.x1 + 5, rect.y1 + 5)
                    page.draw_oval(rect_expanded, color=(1, 0, 0), width=2)
                    
                    annot = page.add_text_annot(rect_expanded.top_right, annot_text)
                    annot.set_colors(stroke=(1, 0, 0) if i_type.lower() == "typo" else (1, 0.5, 0))
                    annot.set_info(content=annot_text, title=f"ข้อผิดพลาด ({i_type})")
                    annot.update()
        
        # วิธีที่ 2 (Fallback): ถ้าหาข้อความไม่เจอ ให้ใช้พิกัดจาก AI
        if not found_coordinates and box and len(box) == 4 and box != [0, 0, 0, 0]:
            ymin, xmin, ymax, xmax = box
            x0 = (xmin / 1000.0) * rect_width
            y0 = (ymin / 1000.0) * rect_height
            x1 = (xmax / 1000.0) * rect_width
            y1 = (ymax / 1000.0) * rect_height
            
            rect = fitz.Rect(x0, y0, x1, y1)
            page.draw_oval(rect, color=(1, 0, 0), width=2)
            
            annot = page.add_text_annot(fitz.Point(x1, y0), annot_text)
            annot.set_colors(stroke=(1, 0, 0) if i_type.lower() == "typo" else (1, 0.5, 0))
            annot.set_info(content=annot_text, title=f"ข้อผิดพลาด ({i_type})")
            annot.update()
            found_coordinates = True
        
        # วิธีที่ 3: แปะ Note ไว้มุมขวาบนกรณีหาพิกัดไม่ได้
        if not found_coordinates:
            annot = page.add_text_annot(fitz.Point(rect_width - 50, 50), annot_text)
            annot.set_colors(stroke=(0, 0, 1))
            annot.set_info(content=annot_text, title="ข้อสังเกต / ภาพรวม")
            annot.update()

uploaded_file = st.file_uploader("เลือกไฟล์แบบแปลน (PDF)", type=["pdf"])

if uploaded_file is not None and model is not None:
    if st.button("🚀 เริ่มการวิเคราะห์ทีละหน้า (แม่นยำสูง)"):
        status_text = st.empty()
        
        try:
            doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
            current_dpi = 150 
            
            for page_num in range(len(doc)):
                status_text.info(f"⏳ กำลังประมวลผลหน้า {page_num + 1} จาก {len(doc)}...")
                
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=current_dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                
                prompt_content = [system_prompt, img]
                
                response = model.generate_content(
                    prompt_content,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                        top_k=1,
                        top_p=0.1,
                    ),
                )
                
                result_json = clean_json_response(response.text)
                
                if result_json:
                    st.markdown(f"### 📄 หน้า {page_num + 1}: {result_json.get('project_type', 'ไม่ระบุ')}")
                    
                    draw_and_annotate_on_pdf(page, result_json.get("issues", []))
                    
                    pix_annotated = page.get_pixmap(dpi=100)
                    img_annotated = Image.open(io.BytesIO(pix_annotated.tobytes("png")))
                    st.image(img_annotated, use_column_width=True)
                    
                    issues_this_page = result_json.get("issues", [])
                    if issues_this_page:
                        with st.expander(f"📝 ข้อเสนอแนะ (หน้า {page_num + 1})", expanded=True):
                            for issue in issues_this_page:
                                i_type = issue.get("type", "Observation")
                                kw = issue.get("error_keyword", "")
                                desc = issue.get("description", "")
                                
                                if i_type.lower() == "typo":
                                    st.error(f"🔤 **[คำผิด: '{kw}']:** {desc}")
                                elif i_type.lower() == "engineering":
                                    st.warning(f"📐 **[วิศวกรรม: '{kw}']:** {desc}")
                                else:
                                    st.info(f"💡 **[ข้อสังเกต]:** {desc}")
                    
                    st.markdown("---")
                
                del pix
                del img
                gc.collect()

            out_pdf_bytes = io.BytesIO()
            doc.save(out_pdf_bytes)
            status_text.success("✅ ตรวจสอบเสร็จสมบูรณ์!")
            
            st.download_button(
                label="📥 ดาวน์โหลดไฟล์ PDF (พร้อมวงกลมสีแดง)",
                data=out_pdf_bytes.getvalue(),
                file_name="Reviewed_Engineering_Drawing_Circles.pdf",
                mime="application/pdf"
            )
                
        except Exception as e:
            status_text.error(f"❌ เกิดข้อผิดพลาด: {e}")
