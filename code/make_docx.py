# -*- coding: utf-8 -*-
"""
make_docx.py —— 将 thesis/paper.md 转换为 Word 版论文（thesis/paper.docx）

不依赖 python-docx：直接按 OOXML 规范手工构造最小可用的 .docx 包
（[Content_Types].xml + _rels/.rels + word/document.xml + word/styles.xml），
支持封面、标题层级、正文、加粗、表格（带边框）、项目符号与等宽代码段。

运行：python code/make_docx.py
"""
import os
import re
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD_PATH = os.path.join(BASE, "thesis", "paper.md")
OUT_PATH = os.path.join(BASE, "thesis", "paper.docx")

COVER_SUB = "—— 以“语调测度—信效度检验—预警模型”五步流水线为例 ——"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def runs_with_bold(text):
    parts = re.split(r"(\*\*.+?\*\*)", text)
    out = []
    for p in parts:
        if not p:
            continue
        if p.startswith("**") and p.endswith("**"):
            out.append('<w:r><w:rPr><w:b/></w:rPr><w:t xml:space="preserve">%s</w:t></w:r>' % esc(p[2:-2]))
        else:
            out.append('<w:r><w:t xml:space="preserve">%s</w:t></w:r>' % esc(p))
    return "".join(out)


def para(text, style=None, bold=False, center=False, size=None, italic=False, color=None):
    props = []
    if style:
        props.append('<w:pStyle w:val="%s"/>' % style)
    rpr = []
    if bold:
        rpr.append("<w:b/>")
    if italic:
        rpr.append("<w:i/>")
    if size:
        rpr.append('<w:sz w:val="%d"/>' % (size * 2))
    if color:
        rpr.append('<w:color w:val="%s"/>' % color)
    if rpr:
        props.append("<w:rPr>%s</w:rPr>" % "".join(rpr))
    if center:
        props.append('<w:jc w:val="center"/>')
    pr = "<w:pPr>%s</w:pPr>" % "".join(props) if props else ""
    return "<w:p>%s%s</w:p>" % (pr, runs_with_bold(text))


def bullet(text):
    return para("• " + text)


def table(rows):
    borders = ('<w:tblBorders>'
               '<w:top w:val="single" w:sz="4" w:color="666666"/>'
               '<w:left w:val="single" w:sz="4" w:color="666666"/>'
               '<w:bottom w:val="single" w:sz="4" w:color="666666"/>'
               '<w:right w:val="single" w:sz="4" w:color="666666"/>'
               '<w:insideH w:val="single" w:sz="4" w:color="666666"/>'
               '<w:insideV w:val="single" w:sz="4" w:color="666666"/>'
               '</w:tblBorders>')
    xml = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>%s</w:tblPr>' % borders]
    for ri, row in enumerate(rows):
        xml.append("<w:tr>")
        for cell in row:
            shd = '<w:shd w:val="clear" w:fill="EFEFEF"/>' if ri == 0 else ""
            xml.append('<w:tc><w:tcPr>%s</w:tcPr><w:p>%s</w:p></w:tc>' % (shd, runs_with_bold(cell)))
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    xml.append("<w:p/>")
    return "".join(xml)


def code_block(lines):
    out = ['<w:p><w:pPr><w:shd w:val="clear" w:fill="F4F4F4"/></w:pPr></w:p>']
    for ln in lines:
        out.append('<w:p><w:pPr><w:shd w:val="clear" w:fill="F4F4F4"/></w:pPr>'
                   '<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/></w:rPr>'
                   '<w:t xml:space="preserve">%s</w:t></w:r></w:p>' % esc(ln))
    out.append("<w:p/>")
    return "".join(out)


def page_break():
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def md_to_docx(md):
    body = []
    lines = md.split("\n")
    i = 0
    first_h1 = True
    after_cover_table = False
    while i < len(lines):
        ln = lines[i].rstrip()
        if ln.startswith("```"):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            body.append(code_block(block))
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                body.append(table(rows))
                if not after_cover_table and first_h1 is False:
                    body.append(page_break())
                    after_cover_table = True
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", ln)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level == 1 and first_h1:
                body.append(para(text, bold=True, center=True, size=26))
                body.append(para(COVER_SUB, center=True, size=13, color="444444"))
                body.append("<w:p/>")
                first_h1 = False
            elif level == 1:
                body.append(para(text, style="Heading1"))
            elif level == 2:
                body.append(para(text, style="Heading1") if not first_h1 else para(text, center=True, size=14))
            elif level == 3:
                body.append(para(text, style="Heading2"))
            else:
                body.append(para(text, style="Heading3"))
            i += 1
            continue
        if ln.strip() in ("---", ""):
            i += 1
            continue
        if ln.startswith(">"):
            text = ln.lstrip("> ").strip()
            if text:
                body.append(para(text, italic=True, color="555555"))
            i += 1
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            body.append(bullet(re.sub(r"^\s*[-*]\s+", "", ln)))
            i += 1
            continue
        body.append(para(ln))
        i += 1
    return "".join(body)


STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Times New Roman" w:eastAsia="宋体" w:hAnsi="Times New Roman"/>
<w:sz w:val="24"/></w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:styleId="Normal" w:default="1"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>
<w:pPr><w:outlineLvl w:val="0"/><w:spacing w:before="240" w:after="120"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>
<w:pPr><w:outlineLvl w:val="1"/><w:spacing w:before="180" w:after="100"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>
<w:pPr><w:outlineLvl w:val="2"/><w:spacing w:before="120" w:after="80"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
</w:styles>"""


def build_docx(body_xml):
    document = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                '<w:body>%s'
                '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
                '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>'
                '</w:body></w:document>' % body_xml)
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                     '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/>'
                     '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                     '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
            '</Relationships>')
    doc_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                '</Relationships>')
    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", STYLES)


def main():
    with open(MD_PATH, encoding="utf-8") as f:
        md = f.read()
    build_docx(md_to_docx(md))
    print("Word 版论文已生成 -> %s（%.1f KB）"
          % (os.path.relpath(OUT_PATH, BASE), os.path.getsize(OUT_PATH) / 1024))


if __name__ == "__main__":
    main()
