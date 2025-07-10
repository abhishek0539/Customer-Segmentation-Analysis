from flask import Flask, request, render_template, send_file, redirect, url_for
import os
from analyse import run_analysis_from_file
from logger import log, get_log_buffer, clear_log_buffer
from chatbot import extract_text_from_pdf, convert_pdf_to_images, generate_insights, generate_report_pdf, format_for_pdf
from datetime import datetime

app = Flask(__name__)

@app.route('/get_logs')
def get_logs():
    return {'logs': get_log_buffer()}


UPLOAD_FOLDER = 'uploads'
REPORT_FOLDER = 'static/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        clear_log_buffer() 
        log("Previous logs cleared.")

        file = request.files['dataset']
        if file and file.filename.endswith('.csv'):
            dataset_path = os.path.join(UPLOAD_FOLDER, 'dataset.csv')
            file.save(dataset_path)
            log("CSV uploaded successfully.")

            summary_path = os.path.join(REPORT_FOLDER, "chatbot_summary.pdf")
            if os.path.exists(summary_path):
                os.remove(summary_path)
                log("🧹 Cleared previous chatbot summary.")

            report_path = os.path.join(REPORT_FOLDER, "report.pdf")
            log("📊 Starting analysis...")
            run_analysis_from_file(dataset_path, report_path)
            log("Analysis complete. Report saved.")

            return redirect(url_for('results'))
        else:
            log("Error: Only CSV files are allowed.")

    return render_template('index.html')



@app.route('/results', methods=['GET'])
def results():
    return render_template('results.html')

@app.route('/download_report')
def download_report():
    return send_file(os.path.join(REPORT_FOLDER, "report.pdf"), as_attachment=True)

@app.route('/chatbot_summary', methods=['GET', 'POST'])
def chatbot_summary():
    pdf_path = os.path.join(REPORT_FOLDER, "report.pdf")
    pdf_text = extract_text_from_pdf(pdf_path)
    images = convert_pdf_to_images(pdf_path)
    insights = generate_insights(pdf_text, images)

    conversation = [{
        "title": "Initial Analysis Summary",
        "content": insights
    }]

    if request.method == 'POST':
        question = request.form.get("question")
        if question:
            from google.generativeai import GenerativeModel
            model = GenerativeModel('gemini-2.5-flash')
            prompt = [
                "As a business analyst, answer this question based on the customer segmentation report:",
                question,
                "\nReport Content:",
                pdf_text,
                "\nPrevious Analysis:",
                insights
            ]
            for img_path in images:
                with open(img_path, "rb") as f:
                    prompt.append({"mime_type": "image/png", "data": f.read()})
            response = model.generate_content(prompt)
            answer = format_for_pdf(response.text)
            conversation.append({
                "title": f"Q: {question}",
                "content": answer
            })

    summary_path = os.path.join(REPORT_FOLDER, "chatbot_summary.pdf")
    generate_report_pdf(conversation, output_path=summary_path)

    return render_template('chatbot.html', insights=insights)


@app.route('/download_summary')
def download_summary():
    return send_file(os.path.join(REPORT_FOLDER, "chatbot_summary.pdf"), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
