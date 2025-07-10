import google.generativeai as genai
import PyPDF2
import os
from pdf2image import convert_from_path
import tempfile
from fpdf import FPDF
from datetime import datetime
import re
from fpdf.enums import XPos, YPos
# import shutil
from logger import log, get_log_buffer

# API Key Configuration
API_KEY = "API-KEY"  # Replace with your actual API key
try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    log(f"Error configuring Gemini API: {e}")
    exit(1)

def check_existing_report():
    """Check if report.pdf already exists in the current directory"""
    if os.path.exists("report.pdf"):
        log("Found existing report.pdf in the current directory")
        choice = input("Would you like to use this file? (yes/no): ").lower()
        if choice in ['y', 'yes']:
            return "report.pdf"
    return None

def get_pdf_file():
    """Prompt user to input PDF file path"""
    existing = check_existing_report()
    if existing:
        return existing

    file_path = input("Enter the full path to your customer segmentation report PDF: ").strip()
    if not os.path.isfile(file_path):
        log("File not found.")
        return None
    return file_path

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception as e:
        log(f"Error reading PDF text: {e}")
        return ""
'''
def convert_pdf_to_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        image_paths = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, image in enumerate(images):
                image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                image.save(image_path, "PNG")
                # ✅ Use shutil.copy for cross-platform compatibility
                new_path = f"./page_{os.path.basename(image_path)}"
                shutil.copy(image_path, new_path)
                image_paths.append(new_path)
        return image_paths
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []
'''

def convert_pdf_to_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        image_paths = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, image in enumerate(images):
                image_path = os.path.join(temp_dir, f"page_{i+1}.png")
                image.save(image_path, "PNG")
                # Replace shutil.copy with manual file copy
                new_path = f"./page_{os.path.basename(image_path)}"
                with open(image_path, 'rb') as src_file:
                    with open(new_path, 'wb') as dest_file:
                        dest_file.write(src_file.read())
                image_paths.append(new_path)
        return image_paths
    except Exception as e:
        log(f"Error converting PDF to images: {e}")
        return []


def clean_text(text):
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    text = text.replace('\u2022', '-')
    text = re.sub(r'\*{2,}', '', text)
    text = re.sub(r'"{2,}', '"', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\&', '&')
    return text.strip()

def format_for_pdf(text):
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        line = clean_text(line)
        if line.startswith('#'):
            level = line.count('#')
            line = line.replace('#', '').strip()
            if level == 1:
                formatted_lines.append(f"\n=== {line.upper()} ===")
            elif level == 2:
                formatted_lines.append(f"\n--- {line.title()} ---")
            elif level == 3:
                formatted_lines.append(f"\n- {line}")
        elif line.startswith('- '):
            formatted_lines.append(line.strip())
        elif re.match(r'^\d+\.\s', line):
            formatted_lines.append(f"\n**{line}**")
        else:
            formatted_lines.append(line)
    return '\n'.join(formatted_lines)

def generate_insights(pdf_text, image_paths):
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        analysis_prompt = """
        You are a senior business analyst examining a customer segmentation report. Provide a comprehensive analysis with:

        1. EXECUTIVE SUMMARY:
        - Key findings from RFM analysis
        - Major customer clusters identified
        - Overall customer behavior patterns

        2. SEGMENT ANALYSIS:
        - For each customer segment/cluster:
          * Demographic characteristics
          * Behavioral patterns
          * Purchase frequency/value
          * Retention potential
        - Comparison between segments

        3. VISUAL ANALYSIS:
        - For each provided chart/graph:
          * Type of visualization
          * Key data points shown
          * Trends/patterns observed
          * Business implications

        4. ACTIONABLE RECOMMENDATIONS:
        - Marketing strategies for each segment
        - Retention improvement tactics
        - Upsell/cross-sell opportunities
        - Budget allocation suggestions

        5. RISK ASSESSMENT:
        - Potential risks in current segmentation
        - Customer groups at risk of churn
        - Data quality concerns

        give data in structured paragraphs and don't use any signs eg. - , *

        Report content:
        """

        prompt = [analysis_prompt, pdf_text]

        if image_paths:
            for idx, image_path in enumerate(image_paths):
                with open(image_path, "rb") as img_file:
                    prompt.append({
                        "mime_type": "image/png",
                        "data": img_file.read()
                    })
                    prompt.append(f"ANALYZE VISUAL {idx+1}: Describe the chart/graph, its purpose, key findings, and business recommendations based on this visualization.")
        else:
            prompt.append("NOTE: No visual elements provided. Focus analysis on textual data.")

        response = model.generate_content(prompt)
        return format_for_pdf(response.text)
    except Exception as e:
        log(f"Error generating insights: {e}")
        return format_for_pdf(f"Analysis Error: {str(e)}")

def safe_multi_cell(pdf, text, font_name="Helvetica", font_size=10, style=""):
    try:
        pdf.set_font(font_name, style, font_size)
        paragraphs = text.split('\n')
        for para in paragraphs:
            if not para.strip():
                pdf.ln(5)
                continue
            clean_para = clean_text(para)
            pdf.multi_cell(0, 5, clean_para.encode('latin-1', 'replace').decode('latin-1'))
            pdf.ln(4)
    except Exception as e:
        log(f"⚠️ Warning: Error adding text to PDF: {e}")
        pdf.set_font(font_name, style, font_size)
        pdf.cell(0, 5, "[Content truncated due to formatting issues]")

def generate_report_pdf(conversation, output_path="chatbot_summary.pdf"):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 20, "CUSTOMER SEGMENTATION ANALYSIS", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(20)
        pdf.set_font('Helvetica', 'I', 10)
        safe_multi_cell(pdf, "This report contains AI-generated business insights based on customer segmentation data. Use findings as strategic recommendations.", 'Helvetica', 10, 'I')

        for entry in conversation:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 12)
            safe_multi_cell(pdf, entry["title"].upper(), 'Helvetica', 12, 'B')
            pdf.ln(8)
            safe_multi_cell(pdf, entry["content"], 'Helvetica', 10, '')

        # ✅ Save directly to path provided
        pdf.output(name=output_path, dest='F')
        log(f"✅ Report generated: {output_path}")
    except Exception as e:
        log(f"Error generating PDF: {e}")


def main_analysis_flow():
    pdf_path = get_pdf_file()
    if not pdf_path:
        return

    log("\nAnalyzing the report...")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        log("No text extracted from PDF")
        return

    image_paths = convert_pdf_to_images(pdf_path)
    log(f"Extracted {len(image_paths)} visual elements")

    conversation = []
    insights = generate_insights(pdf_text, image_paths)
    log("\n📌 Key Insights:")
    log(insights)

    conversation.append({
        "title": "Initial Analysis Findings",
        "content": insights
    })

    while True:
        question = input("\nAsk a follow-up question (or 'exit' to finish): ").strip()
        if question.lower() in ['exit', 'quit', 'done']:
            break

        if not question:
            continue

        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = [
                "As a business analyst, answer this question based on the customer segmentation report:",
                question,
                "\nReport Content:",
                pdf_text,
                "\nPrevious Analysis:",
                insights
            ]
            if image_paths:
                for img_path in image_paths:
                    with open(img_path, "rb") as f:
                        prompt.append({
                            "mime_type": "image/png",
                            "data": f.read()
                        })

            response = model.generate_content(prompt)
            answer = format_for_pdf(response.text)
            log("\nResponse:")
            log(answer)

            conversation.append({
                "title": f"Q: {question}",
                "content": answer
            })

        except Exception as e:
            log(f"Error: {e}")
            conversation.append({
                "title": f"Q: {question}",
                "content": f"Error processing question: {e}"
            })

    generate_report_pdf(conversation)

    for path in image_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    log(f"Could not remove temp image file {path}: {e}")

if __name__ == "__main__":
    log("""
    ====================================
    CUSTOMER SEGMENTATION ANALYSIS TOOL
    ====================================
    """)
    main_analysis_flow()
