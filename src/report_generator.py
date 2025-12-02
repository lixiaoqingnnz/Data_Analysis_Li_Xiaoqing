# src/report_generator.py

import os
import json
import subprocess


# =====================================================================
# 1. Markdown Generation (BMW-style Report + Cover Page + Page Numbers)
# =====================================================================
def generate_markdown(
    report_data_json="reports/temp_report_data.json",
    output_md="reports/final_report.md",
    report_title="BMW Used-Car Analytics Report"
):
    """
    Generate a Markdown report styled for BMW analytics, including:
    - Cover page
    - Page numbers
    - CSS styling for PDF output
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    json_path = os.path.join(project_root, report_data_json)
    md_path = os.path.join(project_root, output_md)
    md_dir = os.path.dirname(md_path)

    # Load report data from JSON
    with open(json_path, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    lines = []

    # =============================================================
    # Inject CSS (WeasyPrint-compatible)
    # =============================================================
    lines.append("<style>")

    # --- Page setup & page numbering ---
    lines.append("@page {")
    lines.append("  margin: 20mm 20mm 25mm 20mm;")
    lines.append("  @bottom-center {")
    lines.append("    content: 'Page ' counter(page) ' of ' counter(pages);")
    lines.append("    font-size: 10px; color: #666;")
    lines.append("  }")
    lines.append("}")
    # First page: no page number
    lines.append("@page:first {")
    lines.append("  @bottom-center { content: ''; }")
    lines.append("}")

    # --- Cover page styles ---
    # --- Cover page styles ---
    lines.append("body { margin: 0; padding: 0; }")

    lines.append(".cover-page {")
    lines.append("  position: absolute;")
    lines.append("  top: 50%;")
    lines.append("  left: 50%;")
    lines.append("  transform: translate(-50%, -50%);")
    lines.append("  text-align: center;")
    lines.append("  width: 80%;")
    lines.append("}")

    lines.append(".cover-title {font-size:40px;font-weight:700;margin-bottom:20px;color:#1C69D4;}")
    lines.append(".cover-subtitle {font-size:20px;margin-bottom:30px;color:#333;}")
    lines.append(".cover-meta {font-size:14px;margin:5px 0;color:#555;}")


    # --- Report content styles ---
    lines.append(".main-title {font-size:32px;font-weight:700;margin-top:40px;margin-bottom:28px;text-align:left;}")
    lines.append(".section-title {background-color:#1C69D4;color:white;padding:10px 14px;font-size:22px;font-weight:700;margin-top:28px;}")
    lines.append(".subsection-title {background-color:#4C8BF5;color:white;padding:8px 12px;font-size:18px;font-weight:600;margin-top:16px;}")
    lines.append(".report-image {width:90%; display:block; margin:10px auto;}")
    lines.append(".narrative {margin-top:4px; text-align:justify;}")
    lines.append("table {border-collapse: collapse; width: 90%; margin: 20px auto;}")
    lines.append("th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}")
    lines.append("th {background-color: #f2f2f2;}")
    lines.append("</style>\n")


    # COVER PAGE
    lines.append('<div class="cover-page">')
    lines.append(f'<div class="cover-title">{report_title}</div>')
    lines.append('<div class="cover-subtitle">Global Used-Car Market Analytics (2020–2024)</div>')
    lines.append('<div class="cover-meta">Prepared by: Automated LLM Analytics Pipeline (Developed by Li Xiaoqing)</div>')
    lines.append('<div class="cover-meta">Date: 2024-12-01</div>')
    lines.append('</div>')

    # Force next page
    lines.append('<div style="page-break-after: always;"></div>\n')

    # MAIN DOCUMENT TITLE (Page 2)
    # lines.append(f'<h1 class="main-title">{report_title}</h1>\n')


    # Build report sections
    section_counter = 0
    subsection_counter = 0
    current_section = None

    for item in report_data:
        # --- Required fields ---
        section = item["section_title"]
        narrative = item["narrative"]

        # --- Optional fields ---
        chart_path = item.get("chart_path", "")
        table_content = item.get("table_content", None)

        # Safely read subsection; ensure None or clean string
        raw_subsection = item.get("subsection_title", None)
        subsection = raw_subsection.strip() if isinstance(raw_subsection, str) else None


        # Section Header (only when section changes)
        if section != current_section:
            section_counter += 1
            subsection_counter = 0
            current_section = section
            lines.append(f'<div class="section-title">{section_counter} {section}</div>')

        # Subsection Header (only if subsection_title exists)
        if subsection:  # Only output if non-empty string
            subsection_counter += 1
            sub_num = f"{section_counter}.{subsection_counter}"
            lines.append(f'<div class="subsection-title">{sub_num} {subsection}</div>')

        # Table insertion (if present)
        if table_content:
            lines.append("\n#### Forecast Segment Comparison (2024 Actual vs 2025 Prediction)\n")
            lines.append('<div style="font-size: 12px;">')
            lines.append(table_content)
            lines.append("</div>\n")

        # Image insertion (if present)
        if chart_path:
            full_chart_path = os.path.join(project_root, chart_path)
            relative_chart_path = os.path.relpath(full_chart_path, md_dir)
            lines.append(f'<img src="{relative_chart_path}" class="report-image"/>')

        # Narrative paragraph
        lines.append(f'<p class="narrative">{narrative}</p>\n')

    # Write MD file
    os.makedirs(md_dir, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown generated: {md_path}")
    return md_path


# =====================================================================
# 2. Markdown → PDF (Pandoc + WeasyPrint)
# =====================================================================
def generate_pdf(md_path="reports/final_report.md", output_pdf="reports/final_report.pdf"):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    md_path = os.path.join(project_root, md_path)
    pdf_path = os.path.join(project_root, output_pdf)
    html_path = md_path.replace(".md", ".html")

    subprocess.run(
        [
            "pandoc",
            md_path,
            "-o",
            html_path,
            "--standalone",
            "--toc",
            "--metadata",
            "title=",
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )

    subprocess.run(
        ["weasyprint", html_path, pdf_path],
        check=True,
        stderr=subprocess.DEVNULL,
    )

    print(f"PDF generated: {pdf_path}")
    return pdf_path


# =====================================================================
# 3. Full Report Pipeline
# =====================================================================
def generate_full_report():
    md_file = generate_markdown()
    generate_pdf(md_file)
    print("Report (MD + PDF) generated successfully!")
