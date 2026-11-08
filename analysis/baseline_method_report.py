"""Render the editable baseline methods Markdown into a five-page PDF."""
import re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT = Path(__file__).resolve().parents[1]
styles = getSampleStyleSheet()
styles['BodyText'].fontSize = 9.5
styles['BodyText'].leading = 13.5
styles['BodyText'].spaceAfter = 6
styles['Heading1'].fontSize = 20
styles['Heading1'].leading = 24
styles['Heading1'].textColor = colors.HexColor('#153951')
styles['Heading2'].fontSize = 12
styles['Heading2'].leading = 15
styles['Heading2'].spaceBefore = 7
styles['Heading2'].spaceAfter = 5
styles['Heading3'].fontSize = 11
styles['Heading3'].leading = 14
styles['Heading3'].spaceBefore = 6
styles['Heading3'].spaceAfter = 4


def markup(text):
    text = escape(text)
    return re.sub(r'https://[^\s]+', lambda m: '<link href="'+m[0]+'" color="#237eaa">'+m[0]+'</link>', text)


story = []
lines = (ROOT/'reports/tfidf_logreg_method.md').read_text().splitlines()
i = 0
while i < len(lines):
    line = lines[i]
    if line == '---PAGE---':
        story.append(PageBreak())
    elif line.startswith('|'):
        data = []
        while i < len(lines) and lines[i].startswith('|'):
            data.append([Paragraph(markup(c.strip()), styles['BodyText']) for c in lines[i].strip('|').split('|')])
            i += 1
        cols = len(data[0])
        widths = {4: [215, 90, 100, 106], 3: [145, 165, 201], 2: [170, 341]}[cols]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff5')),
                              ('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('TOPPADDING',(0,0),(-1,-1),5),
                              ('BOTTOMPADDING',(0,0),(-1,-1),5),
                              ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d4dee5'))]))
        story.extend([t, Spacer(1,7)])
        continue
    elif line.strip():
        level = len(line) - len(line.lstrip('#'))
        style = {1:'Heading1',2:'Heading2',3:'Heading3'}.get(level,'BodyText')
        story.append(Paragraph(markup(line[level:].strip() if level else line),styles[style]))
    i += 1


def footer(canvas, doc):
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#526473'))
    canvas.drawString(42,24,'IE4483 | TF-IDF + Logistic Regression | Method preparation')
    canvas.drawRightString(553,24,str(doc.page))


out = ROOT/'reports/tfidf_logreg_method.pdf'
SimpleDocTemplate(str(out),pagesize=(595.28,841.89),leftMargin=42,rightMargin=42,
                  topMargin=36,bottomMargin=44,title='IE4483 TF-IDF + Logistic Regression',
                  author='Vu Ngoc Linh').build(story,onFirstPage=footer,onLaterPages=footer)
print(out)
