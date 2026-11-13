# -*- coding: utf-8 -*-
"""
将Markdown报告转为HTML，然后用xhtml2pdf生成PDF。
如果xhtml2pdf有黑块问题，会同时生成HTML备用。
"""
import markdown2
import os
import re

base_dir = os.path.dirname(os.path.abspath(__file__))
md_path = os.path.join(base_dir, '大数据应用案例期末项目报告.md')
html_path = os.path.join(base_dir, '大数据应用案例期末项目报告.html')
pdf_path = os.path.join(base_dir, '大数据应用案例期末项目报告.pdf')

with open(md_path, 'r', encoding='utf-8') as f:
    md_text = f.read()

# Remove the "对应评分项" blockquotes entirely (they were AI traces)
md_text = re.sub(r'\n> \*\*对应评分项：.*?\*\*\n', '\n', md_text)

html_body = markdown2.markdown(md_text, extras=['tables', 'fenced-code-blocks', 'header-ids'])

# Clean up any remaining > blockquotes that look like grading markers
html_body = re.sub(r'<blockquote>\s*<p>\s*<strong>对应评分项.*?</p>\s*</blockquote>', '', html_body, flags=re.DOTALL)

html_full = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>大数据应用案例期末项目报告</title>
<style>
@media print {{
    @page {{ size: A4; margin: 2cm 2.2cm; }}
    body {{ font-size: 11pt; }}
    h1 {{ page-break-before: auto; }}
    pre {{ white-space: pre-wrap; word-wrap: break-word; }}
}}
body {{
    font-family: "Microsoft YaHei", "SimSun", "SimHei", "DejaVu Sans", sans-serif;
    font-size: 13px;
    line-height: 1.75;
    color: #222;
    max-width: 880px;
    margin: 0 auto;
    padding: 15px 35px;
}}
h1 {{
    font-size: 21px;
    text-align: center;
    border-bottom: 2.5px solid #1a1a2e;
    padding-bottom: 8px;
    margin-top: 28px;
}}
h2 {{
    font-size: 16px;
    color: #1a1a2e;
    border-bottom: 1px solid #ccc;
    padding-bottom: 4px;
    margin-top: 22px;
}}
h3 {{
    font-size: 14px;
    color: #333;
    margin-top: 16px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 11.5px;
}}
th, td {{
    border: 1px solid #bbb;
    padding: 5px 8px;
    text-align: left;
}}
th {{
    background-color: #e8e8e8;
    font-weight: bold;
}}
tr:nth-child(even) {{
    background-color: #f7f7f7;
}}
code {{
    background-color: #f0f0f0;
    padding: 1px 3px;
    border-radius: 2px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 10.5px;
}}
pre {{
    background-color: #1e1e1e;
    color: #d4d4d4;
    padding: 10px 14px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 10px;
    line-height: 1.45;
}}
pre code {{
    background: none;
    color: #d4d4d4;
    padding: 0;
}}
blockquote {{
    border-left: 3px solid #4a90d9;
    padding: 6px 12px;
    margin: 8px 0;
    background-color: #f0f5fb;
    font-size: 11.5px;
}}
hr {{
    border: none;
    border-top: 1px solid #ccc;
    margin: 18px 0;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

# Save HTML
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_full)
print(f"HTML: {html_path}")

# Generate PDF
try:
    from xhtml2pdf import pisa
    with open(pdf_path, 'wb') as f:
        pisa.CreatePDF(html_full, dest=f, encoding='utf-8')
    print(f"PDF: {pdf_path}")
except Exception as e:
    print(f"PDF generation failed: {e}")
    print("Open the HTML in browser and Ctrl+P to save as PDF.")
