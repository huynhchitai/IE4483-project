"""Generate the descriptive dataset PDF. Run from any directory with Python.

Dependencies: pip install -r analysis/requirements-report.txt
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
OUT.mkdir(exist_ok=True)
train, test = [json.loads((ROOT / 'data' / f'{s}.json').read_text()) for s in ('train', 'test')]
assert all(set(r) == {'reviews', 'sentiments'} for r in train)
assert all(set(r) == {'reviews'} for r in test)
texts = {name: [r['reviews'] for r in rows] for name, rows in [('train', train), ('test', test)]}
assert all(isinstance(s, str) for values in texts.values() for s in values)
assert all(type(r['sentiments']) is int and r['sentiments'] in (0, 1) for r in train)
tokens = lambda s: re.findall(r"[a-z']+", s.lower())
lengths = {name: np.array([len(tokens(s)) for s in values]) for name, values in texts.items()}
labels = Counter(r['sentiments'] for r in train)
counts = {name: Counter(values) for name, values in texts.items()}
overlap = set(texts['train']) & set(texts['test'])
groups = defaultdict(set)
for r in train:
    groups[r['reviews']].add(r['sentiments'])
summary = {'class_counts': dict(labels), 'datasets': {}, 'overlap_unique_texts': len(overlap),
           'overlap_test_rows': sum(s in overlap for s in texts['test']),
           'conflicting_train_texts': sum(len(v) > 1 for v in groups.values())}
for name, rows in [('train', train), ('test', test)]:
    n = lengths[name]
    path = ROOT / 'data' / f'{name}.json'
    summary['datasets'][name] = {
        'rows': len(rows), 'fields': sorted(set().union(*(r.keys() for r in rows))),
        'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'unique_texts': len(counts[name]), 'duplicate_extra_rows': len(rows) - len(counts[name]),
        'empty_reviews': sum(not s.strip() for s in texts[name]),
        'missing_review_fields': sum('reviews' not in r or r['reviews'] is None for r in rows),
        'token_count': int(n.sum()), 'vocabulary_size': len(set(w for s in texts[name] for w in tokens(s))),
        'length': {k: float(v) for k, v in zip(
            ['min', 'mean', 'median', 'p90', 'p95', 'p99', 'max'],
            [n.min(), n.mean(), *np.percentile(n, [50, 90, 95, 99]), n.max()])}}
(OUT / 'dataset_statistics.json').write_text(json.dumps(summary, indent=2) + '\n')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(7, 2.5), layout='constrained')
bars = ax.barh(['Negative (0)', 'Positive (1)'], [labels[0], labels[1]], color=['#e89155', '#237eaa'])
for bar, value in zip(bars, [labels[0], labels[1]]):
    ax.text(value + 70, bar.get_y() + bar.get_height()/2, f'{value:,}  ({value/len(train):.2%})', va='center')
ax.set_xlim(0, 8000)
ax.set_xlabel('Number of training reviews')
fig.savefig(OUT / 'class_balance.png', dpi=170)
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), layout='constrained')
for name, color in [('train', '#237eaa'), ('test', '#e89155')]:
    axes[0].hist(lengths[name], bins=np.arange(0, 321, 10), weights=np.ones(len(lengths[name]))/len(lengths[name])*100,
                 alpha=.6, label=name, color=color)
    x = np.sort(lengths[name])
    axes[1].plot(x, np.arange(1, len(x)+1)/len(x)*100, label=name, color=color)
axes[0].set(xlabel='Tokens per review (0–320 shown)', ylabel='% of all reviews per bin')
axes[1].set(xlabel='Tokens per review (full range)', ylabel='Cumulative % of reviews')
for ax in axes:
    ax.legend(frameon=False)
fig.savefig(OUT / 'review_lengths.png', dpi=170)
plt.close(fig)

styles = getSampleStyleSheet()
styles['Title'].textColor = colors.HexColor('#153951')
styles['BodyText'].fontSize = 10
styles['BodyText'].leading = 15
story = []
def p(s, style='BodyText'):
    story.append(Paragraph(s, styles[style]))
    story.append(Spacer(1, 7))
def table(rows, widths=None):
    cells = [[Paragraph(escape(str(x)), styles['BodyText']) for x in row] for row in rows]
    t = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e6eff5')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'), ('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7), ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d4dee5'))]))
    story.extend([t, Spacer(1, 10)])
def page(title):
    story.append(PageBreak())
    p(title, 'Heading1')

p('Dataset Analysis', 'Title')
p('IE4483 · Product Review Sentiment Classification', 'Heading2')
p('Descriptive analysis | 8 November 2026 | Prepared for Person 1: Data and baseline')
p('1. Dataset size and structure', 'Heading1')
p('The supplied data contains <b>9,252 reviews</b>: 7,401 labelled training examples and 1,851 unlabelled test examples. Each JSON file is an array of records. One row represents one review.')
table([['Property', 'Training set', 'Test set'],
       ['File', 'data/train.json', 'data/test.json'],
       ['Raw table shape', f'{len(train):,} × 2', f'{len(test):,} × 1'],
       ['Columns', 'reviews; sentiments', 'reviews'],
       ['Text input shape', f'({len(train):,},)', f'({len(test):,},)'],
       ['Target shape', f'({len(train):,},)', 'Not provided'],
       ['File size (MiB)', *[f"{summary['datasets'][s]['bytes']/2**20:.3f}" for s in ('train','test')]]], [160, 175, 175])
p('<b>Schema.</b> <font face="Courier">reviews</font> is a text string; <font face="Courier">sentiments</font> is an integer target: 0 = negative and 1 = positive. All loaded records have the expected fields and valid types.' )
p('What does “dimension” mean here?', 'Heading2')
p('The two training columns are one input field and one target field. Raw text has variable length and does not yet have a fixed numerical feature dimension. After TF-IDF, the input matrix will have shape (N, V), where V is the vocabulary learned from the training partition. For a sequence model, token IDs typically have shape (N, L), and embeddings have shape (N, L, D). V, L and D are modelling choices or training-derived values; they are not measured feature dimensions of the raw JSON.')
p('Scope: this report describes the supplied files. It does not train a classifier or use test labels, which are unavailable. File sizes describe disk storage, not model memory or RAM usage.')

page('2. Label distribution')
story.append(Image(str(OUT / 'class_balance.png'), width=7*inch, height=2.5*inch))
ratio = labels[1]/labels[0]
p(f'The positive-to-negative ratio is <b>{ratio:.2f}:1</b>. A classifier that predicts positive for every training example has {labels[1]/len(train):.2%} accuracy on these labels, while its negative-class recall is zero. This is a descriptive reference, not a held-out validation result.')
p('Implications for evaluation', 'Heading2')
p('Report macro-F1 and per-class precision, recall and F1 alongside accuracy and a confusion matrix. Macro-F1 gives the two classes equal weight. Preserve class proportions when creating the shared validation split, while keeping identical review texts in the same partition.')
table([['Class', 'Count', 'Share'], *[[f'{label} = {name}', f'{labels[label]:,}', f'{labels[label]/len(train):.2%}'] for label, name in [(0,'negative'),(1,'positive')]]], [220,140,150])
p('What cannot be inferred', 'Heading2')
p('The test set has no sentiment labels, so its class balance, classification accuracy and F1 cannot be measured from the supplied test file. The two JSON files also contain no star-rating, product ID, reviewer ID or timestamp fields; their original annotation process cannot be verified from these fields alone.')

page('3. Review lengths and vocabulary')
p('For reproducibility, a token is a match of <font face="Courier">[a-z\']+</font> after lowercasing, matching the existing EDA script. This is a simple descriptive tokenizer, not a neural model tokenizer: digits and non-ASCII letters are excluded, and apostrophes are retained. Counts depend on this definition.')
table([['Length statistic (tokens)', 'Training', 'Test'],
       *[[key.upper() if key.startswith('p') else key.capitalize(), *[f"{summary['datasets'][s]['length'][key]:,.1f}" for s in ('train','test')]] for key in ['min','mean','median','p90','p95','p99','max']],
       ['Total token occurrences', *[f"{summary['datasets'][s]['token_count']:,}" for s in ('train','test')]],
       ['Distinct tokens', *[f"{summary['datasets'][s]['vocabulary_size']:,}" for s in ('train','test')]]], [230,140,140])
story.append(Image(str(OUT / 'review_lengths.png'), width=7*inch, height=2.8*inch))
p('The distributions have a long right tail: most reviews are short, but a small number are much longer. The histogram shows lengths up to 320 tokens; its percentages use all reviews as the denominator. The cumulative plot includes the full range. Percentiles use NumPy linear interpolation.')
coverage = {limit: float(np.mean(lengths['train'] <= limit))*100 for limit in (150,200)}
p(f'Lengths of 150 and 200 tokens cover {coverage[150]:.1f}% and {coverage[200]:.1f}% of training reviews respectively under this tokenizer. These are initial sequence-length candidates; measure coverage again with the actual model tokenizer. Descriptive full-corpus vocabulary counts must not be used to fit a validation vocabulary.')

page('4. Data quality and modelling implications')
table([['Quality check', 'Training', 'Test'],
       ['Missing/null review fields', *[summary['datasets'][s]['missing_review_fields'] for s in ('train','test')]],
       ['Empty / whitespace-only reviews', *[summary['datasets'][s]['empty_reviews'] for s in ('train','test')]],
       ['Unique exact review texts', *[f"{summary['datasets'][s]['unique_texts']:,}" for s in ('train','test')]],
       ['Extra duplicate rows', *[summary['datasets'][s]['duplicate_extra_rows'] for s in ('train','test')]],
       ['Texts repeated at least twice', *[sum(v>1 for v in counts[s].values()) for s in ('train','test')]]], [270,120,120])
p(f"There are <b>{summary['conflicting_train_texts']} training texts with conflicting labels</b>. Train/test overlap contains <b>{len(overlap)} distinct exact texts</b>, covering <b>{summary['overlap_test_rows']} test rows</b>. Exact equality preserves case, punctuation and whitespace; near-duplicates are not assessed. Duplicate counts represent extra copies beyond the first occurrence.")
with (ROOT / 'data/submission_template.csv').open(newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    template = list(reader)
aligned = [r['reviews'] for r in template] == texts['test']
p('Submission file check', 'Heading2')
p(f'The supplied submission template has {len(template):,} rows and columns {escape(str(fields))}. Exact review text and row order match test.json: <b>{aligned}</b>. Its predictions are placeholders and provide no test ground truth. The unrelated sampleSubmission.csv is excluded from this analysis.')
p('Recommended next steps', 'Heading2')
for text in [
    'Create and save one shared validation split with a fixed seed. Group exact duplicate texts before splitting, and check the resulting class proportions.',
    'Fit TF-IDF or a neural vocabulary on the training partition only. Preserve negation during preprocessing and evaluate any cleaning choices on validation.',
    'Use the same held-out split for the TF-IDF and GRU comparison. Record feature shape, sparse/dense storage, runtime and measured memory after implementation.',
    'Keep the supplied test order intact when exporting predictions. Treat overlap with training as a limitation when interpreting eventual test performance.']:
    p('• ' + text)
p('Reproduction and source', 'Heading2')
p('Source: local data/train.json, data/test.json and data/submission_template.csv from the cloned IE4483-project repository. Run <font face="Courier">.venv/bin/python analysis/dataset_report.py</font>. Exact statistics and input SHA-256 fingerprints are saved in reports/dataset_statistics.json. No external provenance claims or model results are assumed.')

def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#d4dee5'))
    canvas.line(42, 36, 553, 36)
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#526473'))
    canvas.drawString(42, 24, 'IE4483 | Dataset analysis | 08 Nov 2026')
    canvas.drawRightString(553, 24, str(doc.page))

pdf = OUT / 'dataset_analysis.pdf'
SimpleDocTemplate(str(pdf), pagesize=(595.28,841.89), rightMargin=42,leftMargin=42,
                  topMargin=38,bottomMargin=48, title='IE4483 Dataset Analysis',
                  author='Vu Ngoc Linh').build(story, onFirstPage=footer, onLaterPages=footer)
print(json.dumps(summary, indent=2))
print(f'PDF: {pdf}')
