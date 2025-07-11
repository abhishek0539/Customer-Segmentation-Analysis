# Customer-Segmentation-Analysis

# Customer Segmentation & AI-Powered Business Insights

This project is an end-to-end system for conducting customer segmentation using transactional data and generating analytical reports and business insights powered by Google Gemini AI. The system supports PDF report generation, automated summarization, and natural language Q&A functionality.

[Click here to watch demo video]([https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing](https://drive.google.com/file/d/1JgmYVtG43oeIy0nJablAso2rkaKxR33W/view?usp=sharing))

## Project Overview

This platform allows analysts or business users to:
- Upload transactional datasets in CSV format.
- Automatically generate detailed analytical reports.
- Summarize findings using generative AI.
- Ask follow-up questions and receive intelligent responses.
- Download both the standard and AI-enhanced summary reports.
- Use included sample datasets to test the tool.

## Functional Workflow

### 1. Dataset Upload
Users upload a CSV file containing transactional data with typical columns such as:
- Customer ID
- Invoice Date
- Net Amount or Transaction Value
- Gender, Age, and other optional demographic information

> Note: A few sample transactional datasets are included in the project to help new users quickly test and explore the tool without needing to prepare custom data.

### 2. Automated Report Generation
Once uploaded, the system generates a multi-page `report.pdf` via `analyse.py`, which includes:

**RFM Analysis:**
- Recency: Time since last transaction
- Frequency: Total number of transactions
- Monetary: Total customer spend

**Visual Outputs:**
- Histograms for R, F, M distributions
- Correlation heatmaps
- Elbow plot to determine optimal clusters
- PCA-reduced scatter plots for clusters
- Bar charts for product category frequency
- Demographic comparisons (e.g., gender, age)

**Customer Clustering:**
- Uses K-Means clustering based on RFM or selected features
- Segments customers into actionable groups

**Behavioral Segmentation:**
- Identifies personas like loyal customers, new buyers, dormant users, or at-risk segments

## AI-Powered Summary Generation

Using Google Gemini via the `chatbot.py` module, the system:
- Extracts content and visuals from `report.pdf`
- Generates a structured textual summary including:
  - Executive summary
  - Segment-wise breakdown
  - Visual interpretation
  - Strategic recommendations
  - Risk assessment

The AI-generated content is formatted and saved into `chatbot_summary.pdf`.

## Follow-Up Q&A

After reading the summary, users can optionally:
- Submit custom business-related questions
- Receive detailed answers based on the uploaded dataset and insights
- Append the Q&A to the chatbot summary PDF

This makes it suitable for decision-makers to explore the data further through natural language.

## Report Downloads

The following reports can be downloaded:
- `report.pdf`: Standard analysis generated using data science models and visualizations
- `chatbot_summary.pdf`: AI-generated business summary and optional Q&A content

Each report is refreshed for every new dataset upload. Previous summaries are automatically removed to avoid confusion.

## Technical Stack

|    Component     |         Technology            |
|------------------|-------------------------------|
| Backend Web App  | Flask                         |
| Visualization    | Matplotlib, Seaborn           |
| PDF Generation   | FPDF2                         |
| OCR/Image Tools  | PyPDF2, pdf2image, Pillow     |
| AI Integration   | Google Generative AI (Gemini) |

## Folder Structure

```
Cust_segmentation/
├── app.py                  # Main Flask app
├── analyse.py              # Data analysis and report generation
├── chatbot.py              # AI summary and Q&A handling
├── logger.py               # Log tracking
├── templates/              # HTML interface templates
├── static/reports/         # Output PDF files
├── uploads/                # Uploaded CSVs
├── test_datasets/          # Sample transactional datasets for testing
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

## Setup Instructions

### 1. Install dependencies:
pip install -r requirements.txt

### 2. Install Poppler (for PDF image conversion):
- Windows: [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/) and add to PATH
- Linux: `sudo apt install poppler-utils`
- macOS: `brew install poppler`

### 3. Run the Flask application:
python app.py

Access the application at: `http://127.0.0.1:5000`

## Sample Use Case

1. Upload a transactional dataset or use one from `test_datasets/`.
2. The system automatically generates a PDF report (`report.pdf`) containing:
   - Cluster visuals
   - Demographic segmentation
   - Behavioral insights

3. Visit the chatbot interface to generate an AI summary (`chatbot_summary.pdf`).
4. Optionally ask business questions such as:
   - “What actions should be taken for low-frequency buyers?”
   - “How do male and female customers differ in Cluster 2?”
5. Download the enhanced report for presentation or decision-making.
