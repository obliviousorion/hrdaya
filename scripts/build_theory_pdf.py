#!/usr/bin/env python3
"""Build a publication-quality PDF from docs/THEORY_DUMP_ML_CONCEPTS.md.

Uses latex2mathml for KaTeX/LaTeX math conversion, markdown for GFM tables
and fenced code, WeasyPrint for CSS paged media PDF rendering, and Ghostscript
for print-grade compression ensuring the final PDF stays strictly under 1 MB.
"""
import os
import re
import subprocess
from pathlib import Path
import markdown
from latex2mathml.converter import convert as latex2mathml_convert
import weasyprint

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs/THEORY_DUMP_ML_CONCEPTS.md"
PDF_PATH = ROOT / "docs/THEORY_DUMP_ML_CONCEPTS.pdf"
LOGO_PATH = ROOT / "poster/bits_logo.png"


def convert_math_to_mathml(text: str) -> str:
    """Find $$...$$ and $...$ and convert them to MathML using latex2mathml."""
    # Convert block math $$...$$
    def replace_block(match):
        latex = match.group(1).strip()
        latex = latex.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
        latex = latex.replace(r"\quad", " ")
        try:
            mml = latex2mathml_convert(latex)
            return f'<div class="math-block">{mml}</div>'
        except Exception:
            return f'<div class="math-block fallback-math"><code>{latex}</code></div>'

    text = re.sub(r"\$\$(.*?)\$\$", replace_block, text, flags=re.DOTALL)

    # Convert inline math $...$
    def replace_inline(match):
        latex = match.group(1).strip()
        latex = latex.replace(r"\quad", " ")
        try:
            mml = latex2mathml_convert(latex)
            return f'<span class="math-inline">{mml}</span>'
        except Exception:
            return f'<code class="math-inline-fallback">{latex}</code>'

    # Match $...$ where $ is not preceded or followed by space or digit
    text = re.sub(r"(?<!\$)\$([^\$\n]+?)\$(?!\$)", replace_inline, text)
    return text


def enhance_html_elements(html: str) -> str:
    """Style images into figures and enhance blockquotes with callout icons."""
    # Convert paragraph images and adjacent italics into <figure> and <figcaption>
    def replace_figure(match):
        img_src = match.group(1)
        img_alt = match.group(2)
        caption = match.group(3)
        return (
            f'<figure class="report-figure">\n'
            f'  <img src="{img_src}" alt="{img_alt}" />\n'
            f'  <figcaption>{caption}</figcaption>\n'
            f'</figure>'
        )

    # Matches <p><img src="..." alt="..." />\n<em>...</em></p>
    html = re.sub(
        r'<p><img[^>]*src="([^"]+)"[^>]*alt="([^"]*)"[^>]*\s*\/>\s*<em>(.*?)<\/em><\/p>',
        replace_figure,
        html,
        flags=re.DOTALL
    )

    # Style blockquotes as highlighted callout boxes
    def style_callout(match):
        title = match.group(1)
        body = match.group(2)
        return (
            f'<blockquote class="callout-box">\n'
            f'  <div class="callout-title">💡 {title}</div>\n'
            f'  <div class="callout-body">{body}</div>\n'
            f'</blockquote>'
        )

    html = re.sub(
        r'<blockquote>\s*<p><strong>([^:<]+?):?<\/strong>\s*(.*?)<\/p>\s*<\/blockquote>',
        style_callout,
        html,
        flags=re.DOTALL
    )

    return html


def build_pdf():
    print(f"Reading markdown from {MD_PATH}...")
    content = MD_PATH.read_text(encoding="utf-8")

    # Math conversion first
    print("Converting LaTeX math to MathML...")
    content_math = convert_math_to_mathml(content)

    # Convert Markdown to HTML
    print("Parsing Markdown to HTML with tables & code extensions...")
    html_body = markdown.markdown(
        content_math,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
            "attr_list",
        ],
    )

    # Enhance figures, callouts, and layout
    html_body = enhance_html_elements(html_body)

    # Wrap in full HTML document with professional CSS paged media
    logo_uri = LOGO_PATH.as_uri() if LOGO_PATH.exists() else ""
    logo_html = f'<img src="{logo_uri}" class="header-logo" alt="BITS Pilani" />' if logo_uri else ""

    css = """
    @page {
        size: A4 portrait;
        margin: 22mm 18mm 22mm 18mm;
        @top-left {
            content: "MLEE Capstone Project · ECG Biometrics";
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 8pt;
            font-weight: 600;
            color: #1e3a8a;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        @top-right {
            content: "ML Theory & Viva Study Guide";
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 8pt;
            color: #64748b;
            font-style: italic;
        }
        @bottom-left {
            content: "BITS Pilani · Machine Learning for Edge Engineering";
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 8pt;
            color: #94a3b8;
        }
        @bottom-right {
            content: "Page " counter(page) " of " counter(pages);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 8pt;
            font-weight: 600;
            color: #1e3a8a;
        }
    }

    @page:first {
        @top-left { content: none; }
        @top-right { content: none; }
    }

    * {
        box-sizing: border-box;
    }

    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 9.8pt;
        line-height: 1.55;
        color: #1e293b;
        background-color: #ffffff;
    }

    .title-banner {
        border-bottom: 3px solid #1e3a8a;
        padding-bottom: 15px;
        margin-bottom: 25px;
    }

    .header-logo {
        height: 55px;
        float: right;
        margin-bottom: 10px;
    }

    h1.doc-title {
        font-size: 21pt;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }

    .doc-subtitle {
        font-size: 11pt;
        font-weight: 500;
        color: #1e40af;
        margin: 0 0 10px 0;
    }

    .doc-meta {
        font-size: 8.8pt;
        color: #64748b;
        margin: 0;
    }

    h1, h2, h3, h4 {
        color: #0f172a;
        font-weight: 700;
        page-break-after: avoid;
        margin-top: 1.4em;
        margin-bottom: 0.5em;
    }

    h1 {
        font-size: 15pt;
        border-bottom: 1.5px solid #e2e8f0;
        padding-bottom: 5px;
        color: #1e3a8a;
    }

    h2 {
        font-size: 12.8pt;
        color: #1e40af;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 3px;
    }

    h3 {
        font-size: 10.8pt;
        color: #1e293b;
    }

    h4 {
        font-size: 9.8pt;
        color: #475569;
    }

    p {
        margin: 0 0 0.8em 0;
        text-align: justify;
    }

    strong {
        font-weight: 600;
        color: #0f172a;
    }

    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 18px 0;
    }

    /* Code and ASCII diagrams */
    pre {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 9px 13px;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 8.0pt;
        line-height: 1.35;
        color: #0f172a;
        overflow-x: hidden;
        page-break-inside: avoid;
        margin: 12px 0;
        white-space: pre-wrap;
        word-break: break-word;
    }

    code {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 8.6pt;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1px 4px;
        border-radius: 3px;
    }

    pre code {
        background: transparent;
        padding: 0;
        font-size: 8.0pt;
    }

    /* Math styling */
    .math-block {
        display: block;
        text-align: center;
        margin: 9px 0;
        overflow-x: auto;
        page-break-inside: avoid;
        font-size: 10.2pt;
    }

    .math-inline {
        display: inline;
        font-size: 9.8pt;
    }

    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 8.6pt;
        page-break-inside: avoid;
    }

    th, td {
        padding: 6px 9px;
        border: 1px solid #cbd5e1;
        text-align: left;
    }

    th {
        background-color: #1e3a8a;
        color: #ffffff;
        font-weight: 600;
        font-size: 8.8pt;
    }

    tr:nth-child(even) {
        background-color: #f8fafc;
    }

    tr:hover {
        background-color: #f1f5f9;
    }

    /* Blockquotes & Callouts */
    blockquote {
        margin: 12px 0;
        padding: 10px 14px;
        border-left: 4px solid #3b82f6;
        background-color: #eff6ff;
        border-radius: 0 6px 6px 0;
        font-size: 9.2pt;
        color: #1e293b;
        page-break-inside: avoid;
    }

    blockquote.callout-box {
        border-left: 4px solid #16a34a;
        background-color: #f0fdf4;
    }

    .callout-title {
        font-weight: 700;
        color: #15803d;
        margin-bottom: 4px;
        font-size: 9.5pt;
    }

    .callout-body {
        color: #166534;
    }

    /* Figures */
    figure.report-figure {
        margin: 16px 0;
        text-align: center;
        page-break-inside: avoid;
        background-color: #ffffff;
        padding: 8px;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
    }

    figure.report-figure img {
        max-width: 100%;
        height: auto;
        border-radius: 4px;
        display: block;
        margin: 0 auto;
    }

    figcaption {
        font-size: 8.4pt;
        color: #475569;
        margin-top: 8px;
        line-height: 1.4;
        text-align: center;
        padding: 0 8px;
        font-style: normal;
    }

    /* Lists */
    ul, ol {
        margin: 0 0 0.8em 0;
        padding-left: 20px;
    }

    li {
        margin-bottom: 4px;
    }
    """

    full_html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>ECG Biometrics: Comprehensive ML Theory & Defense Study Guide</title>
      <style>{css}</style>
    </head>
    <body>
      <div class="title-banner">
        {logo_html}
        <h1 class="doc-title">Mastering ECG Biometrics</h1>
        <div class="doc-subtitle">From Biological Waveforms to Hyperspherical Metric Learning on Microcontrollers</div>
        <p class="doc-meta"><strong>Course:</strong> MLEE (Machine Learning for Edge Engineering) · <strong>Institution:</strong> BITS Pilani · <strong>Platform:</strong> RP2040 Cortex-M0+ · <strong>Document:</strong> Technical Defense &amp; ML Theory Study Guide</p>
      </div>
      {html_body}
    </body>
    </html>
    """

    raw_pdf_path = PDF_PATH.with_suffix(".tmp.pdf")
    print(f"Generating initial PDF with WeasyPrint at {raw_pdf_path}...")
    doc = weasyprint.HTML(string=full_html, base_url=str(ROOT))
    doc.write_pdf(str(raw_pdf_path))

    # Ghostscript optimization to keep print quality while ensuring file stays < 1 MB
    print(f"Applying print-grade PDF stream compression...")
    gs_cmd = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        "-dPDFSETTINGS=/printer",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={PDF_PATH}",
        str(raw_pdf_path),
    ]
    subprocess.run(gs_cmd, check=True)
    if raw_pdf_path.exists():
        raw_pdf_path.unlink()

    size_bytes = PDF_PATH.stat().st_size
    size_kb = size_bytes / 1024
    print(f"PDF successfully created: {PDF_PATH} ({size_kb:.1f} KB / {size_bytes} bytes)")
    if size_bytes < 1_000_000:
        print("PASS: PDF is strictly under the 1 MB Git pre-commit threshold.")
    else:
        print("WARNING: PDF exceeds 1 MB limit!")


if __name__ == "__main__":
    build_pdf()
