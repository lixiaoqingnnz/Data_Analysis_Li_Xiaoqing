# src/report_generator.py

import os
import json
import subprocess


# =====================================================================
# 🎨 1. Generate Markdown (BMW-style report)
# =====================================================================
def generate_markdown(
    report_data_json="reports/temp_report_data.json",
    output_md="reports/final_report.md",
    report_title="BMW Used-Car Analytics Report"
):
    """
    Generate a BMW-styled Markdown report with:
    - BMW blue section headers
    - Auto-numbering (1, 1.1, 1.2...)
    - Unified image sizes
    - CSS styles for PDF output
    """

    # --- Detect project root ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    json_path = os.path.join(project_root, report_data_json)
    md_path = os.path.join(project_root, output_md)
    md_dir = os.path.dirname(md_path)

    # Load report data
    with open(json_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    lines = []

    # Inject CSS (WeasyPrint uses this fully!)
    lines.append("<style>")
    lines.append(".section-title {background-color:#1C69D4;color:white;padding:10px 14px;font-size:22px;font-weight:700;margin-top:28px;}")
    lines.append(".subsection-title {background-color:#4C8BF5;color:white;padding:8px 12px;font-size:18px;font-weight:600;margin-top:16px;}")
    lines.append(".report-image {width:90%; display:block; margin:10px auto;}")
    lines.append(".narrative {margin-top:4px; text-align:justify;}")
    # 新增表格样式（可选，但推荐）
    lines.append("table {border-collapse: collapse; width: 90%; margin: 20px auto;}")
    lines.append("th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}")
    lines.append("th {background-color: #f2f2f2;}")
    lines.append("</style>\n")

    # Title
    lines.append(f"# {report_title}\n")

    # Section/subsection numbering
    section_counter = 0
    subsection_counter = 0
    current_section = None

    # Build Markdown
    for item in report_data:
        section = item["section_title"]
        subsection = item["subsection_title"]
        narrative = item["narrative"]
        chart_path = item.get("chart_path", "") # 使用 .get() 确保键存在
        table_content = item.get("table_content", None) # 获取表格内容

        # SECTION
        if section != current_section:
            section_counter += 1
            subsection_counter = 0
            current_section = section
            lines.append(f'<div class="section-title">{section_counter} {section}</div>')

        # SUBSECTION
        subsection_counter += 1
        sub_num = f"{section_counter}.{subsection_counter}"
        lines.append(f'<div class="subsection-title">{sub_num} {subsection}</div>')
        
        # TABLE (Placement for Section 4)
        if table_content:
            lines.append("\n#### Forecast Segment Comparison (2024 Actual vs 2025 Prediction)\n")

            # 🔽 在这里包一层 div，缩小表格字体
            lines.append('<div style="font-size: 12px;">')
            lines.append(table_content)  # 原来的表格内容
            lines.append('</div>')

            lines.append("\n")
        
        # IMAGE (Only render if path exists)
        if chart_path:
            # Resolve relative path for images
            full_chart_path = os.path.join(project_root, chart_path)
            relative_chart_path = os.path.relpath(full_chart_path, md_dir)
            lines.append(f'<img src="{relative_chart_path}" class="report-image"/>')

        # NARRATIVE（用 <p> 包起来）
        lines.append(f'<p class="narrative">{narrative}</p>\n')

    # Save MD
    os.makedirs(md_dir, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown generated: {md_path}")
    return md_path


# =====================================================================
# 📄 2. Convert Markdown → PDF (via WeasyPrint)
# =====================================================================
def generate_pdf(md_path="reports/final_report.md", output_pdf="reports/final_report.pdf"):
    """
    Convert a CSS-enhanced Markdown file to PDF using:
      Markdown → HTML (pandoc)
      HTML → PDF (weasyprint)
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    md_path = os.path.join(project_root, md_path)
    pdf_path = os.path.join(project_root, output_pdf)
    html_path = md_path.replace(".md", ".html")

    # Step 1: Convert md → html
    subprocess.run([
        "pandoc", md_path,
        "-o", html_path,
        "--standalone",
        "--toc"
    ], check=True, stderr=subprocess.DEVNULL)

    # Step 2: Convert html → pdf via WeasyPrint
    subprocess.run([
        "weasyprint",
        html_path,
        pdf_path
    ], check=True, stderr=subprocess.DEVNULL)

    print(f"✔ PDF generated: {pdf_path}")
    return pdf_path



# =====================================================================
# 🎯 3. MASTER FUNCTION — Call this from main.py
# =====================================================================
def generate_full_report():
    """
    One-click full report pipeline:
    1. Read temp_report_data.json
    2. Generate Markdown
    3. Generate PDF
    """
    md_file = generate_markdown()
    generate_pdf(md_file)
    print("Report (MD + PDF) generated successfully!")
