"""Train-only analysis and a diagnostic split simulation; no split indices exported."""
import hashlib
import json
import random
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
rows = json.loads((ROOT / 'data/train.json').read_text())
assert all(set(r) == {'reviews', 'sentiments'} and isinstance(r['reviews'], str)
           and type(r['sentiments']) is int and r['sentiments'] in (0, 1) for r in rows)
texts = [r['reviews'] for r in rows]
assert all(s.strip() for s in texts)
y = np.array([r['sentiments'] for r in rows])
n = np.array([len(re.findall(r"[a-z']+", s.lower())) for s in texts])
groups = defaultdict(list)
normalized = defaultdict(list)
for i, s in enumerate(texts):
    groups[s].append(i)
    normalized[' '.join(s.casefold().split())].append(i)
conflicts = lambda gs: sum(len({int(y[i]) for i in ids}) > 1 for ids in gs.values())
assert conflicts(groups) == 0
sizes = Counter(len(ids) for ids in groups.values())

# Diagnostic only: an independently shuffled, per-class 80/20 row split.
naive_val = []
for label in (0, 1):
    ids = np.flatnonzero(y == label).tolist()
    random.Random(42 + label).shuffle(ids)
    naive_val.extend(ids[:round(.2 * len(ids))])

def describe(val):
    val = set(val)
    tr = set(range(len(rows))) - val
    trtexts, vtexts = {texts[i] for i in tr}, {texts[i] for i in val}
    return {'train_rows': len(tr), 'validation_rows': len(val),
            'train_class_counts': dict(Counter(int(y[i]) for i in tr)),
            'validation_class_counts': dict(Counter(int(y[i]) for i in val)),
            'overlapping_exact_texts': len(trtexts & vtexts),
            'validation_rows_seen_in_train': sum(texts[i] in trtexts for i in val),
            'train_mean_length': float(n[list(tr)].mean()),
            'validation_mean_length': float(n[list(val)].mean())}

# Grouped candidate: shuffle exact-text groups separately per class, then select
# the prefix whose row total is closest to 20% of that class. No row is discarded.
candidate = []
for label in (0, 1):
    gs = sorted([ids for ids in groups.values() if y[ids[0]] == label], key=lambda ids: ids[0])
    random.Random(42 + label).shuffle(gs)
    totals = np.cumsum([0] + [len(ids) for ids in gs])
    k = min(range(len(totals)), key=lambda j: abs(totals[j] - .2 * int((y == label).sum())))
    candidate.extend(i for ids in gs[:k] for i in ids)
naive, grouped = describe(naive_val), describe(candidate)
assert grouped['overlapping_exact_texts'] == 0
assert grouped['train_rows'] + grouped['validation_rows'] == len(rows)

stats = {'source_sha256': hashlib.sha256((ROOT/'data/train.json').read_bytes()).hexdigest(),
         'rows': len(rows), 'exact_unique_texts': len(groups),
         'exact_group_size_distribution': dict(sorted(sizes.items())),
         'normalized_unique_texts': len(normalized), 'normalized_conflicting_groups': conflicts(normalized),
         'row_split_simulation': naive, 'exact_group_split_simulation': grouped,
         'classes': {}}
for label in (0, 1):
    a = n[y == label]
    gs = [ids for ids in groups.values() if y[ids[0]] == label]
    stats['classes'][label] = {'rows': len(a), 'unique_texts': len(gs),
        'extra_duplicate_rows': len(a)-len(gs), 'mean_length': float(a.mean()),
        'median_length': float(np.median(a)), 'p90': float(np.percentile(a,90)),
        'p95': float(np.percentile(a,95)), 'max_length': int(a.max()),
        'over_150_tokens': int((a>150).sum()), 'over_200_tokens': int((a>200).sum())}
(OUT/'train_split_statistics.json').write_text(json.dumps(stats, indent=2)+'\n')

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1,2,figsize=(7,2.7),layout='constrained')
for label, name, color in [(0,'Negative','#e89155'),(1,'Positive','#237eaa')]:
    a = np.sort(n[y==label])
    axes[0].plot(a,np.arange(1,len(a)+1)/len(a)*100,label=name,color=color)
axes[0].set(xlabel='Review length (tokens; 0–350 shown)',ylabel='Cumulative % within class',xlim=(0,350))
axes[0].legend(frameon=False)
axes[1].bar([str(k) for k in sorted(sizes)], [sizes[k] for k in sorted(sizes)],color='#237eaa')
axes[1].set(xlabel='Rows per exact-text group',ylabel='Number of groups (log scale)',yscale='log')
for j,k in enumerate(sorted(sizes)):
    axes[1].text(j,sizes[k]*1.15,str(sizes[k]),ha='center',fontsize=9)
axes[1].set_ylim(.7,max(sizes.values())*4)
fig.savefig(OUT/'train_split_diagnostics.png',dpi=180)
plt.close(fig)

styles = getSampleStyleSheet()
styles['BodyText'].fontSize=9.5
styles['BodyText'].leading=13
styles['Heading2'].fontSize=12
styles['Heading2'].spaceBefore=8
styles['Title'].textColor=colors.HexColor('#153951')
story=[]
def p(text,style='BodyText'):
    story.extend([Paragraph(text,styles[style]),Spacer(1,5)])
def table(data,widths):
    t=Table([[Paragraph(escape(str(v)),styles['BodyText']) for v in row] for row in data],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff5')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d4dee5'))]))
    story.extend([t,Spacer(1,10)])
def page(title):
    story.append(PageBreak());p(title,'Heading1')

p('Training Set Analysis','Title')
p('IE4483 | Preparing the shared train/validation split','Heading2')
p('8 November 2026 | Train-only descriptive analysis and split diagnostics')
p('1. Class structure and review lengths','Heading1')
p('Only data/train.json is used. The supplied training pool contains <b>7,401 rows and 2 columns</b>: one raw-text input and one binary sentiment target. There are no missing/null fields, empty reviews or invalid labels. The official test set is excluded from every statistic and simulation in this report.')
cs=stats['classes']
table([['Statistic','Negative (0)','Positive (1)'],
       ['Rows',*[f"{cs[k]['rows']:,}" for k in (0,1)]],
       ['Share of training pool',*[f"{cs[k]['rows']/len(rows):.2%}" for k in (0,1)]],
       ['Unique exact texts',*[f"{cs[k]['unique_texts']:,}" for k in (0,1)]],
       ['Extra duplicate rows',*[cs[k]['extra_duplicate_rows'] for k in (0,1)]],
       *[[title,*[f"{cs[k][key]:,.1f}" for k in (0,1)]] for title,key in [('Mean length','mean_length'),('Median length','median_length'),('90th percentile','p90'),('95th percentile','p95'),('Maximum length','max_length')]],
       ['Over 150 tokens',*[f"{cs[k]['over_150_tokens']} ({cs[k]['over_150_tokens']/cs[k]['rows']:.1%})" for k in (0,1)]]],[230,140,140])
story.append(Image(str(OUT/'train_split_diagnostics.png'),width=504,height=194.4))
p('Token definition: lowercase text, then regex [a-z\']+. Lengths are descriptive counts, not GRU/subword token counts. Percentiles use linear interpolation. Negative reviews have a different length distribution; check length by class after splitting rather than checking class counts alone.')

page('2. Duplicate groups and leakage risk')
p(f'There are <b>{len(groups):,} unique exact texts</b> and <b>{len(rows)-len(groups)} extra duplicate rows</b>. Of these groups, {sum(count for size,count in sizes.items() if size>1)} contain multiple rows. No exact-text group has conflicting sentiment labels. The largest group contains {max(sizes)} rows.')
table([['Group size (rows)','Number of groups','Rows represented'],*[[size,count,size*count] for size,count in sorted(sizes.items())]],[175,170,165])
p('Why ordinary stratification is insufficient','Heading2')
p('Stratification preserves the label ratio, but independent row assignment can place copies of the same review on both sides. Validation can then reward memorisation. Keep all rows sharing an exact review string together when constructing the shared split.')
p('Diagnostic comparison: 80/20, seed 42','Heading2')
table([['Check','Row-level split','Exact-text grouped split'],
       ['Training rows',naive['train_rows'],grouped['train_rows']],
       ['Validation rows',naive['validation_rows'],grouped['validation_rows']],
       ['Validation negative / positive',f"{naive['validation_class_counts'].get(0,0)} / {naive['validation_class_counts'].get(1,0)}",f"{grouped['validation_class_counts'].get(0,0)} / {grouped['validation_class_counts'].get(1,0)}"],
       ['Shared exact texts',naive['overlapping_exact_texts'],grouped['overlapping_exact_texts']],
       ['Validation rows seen in training',naive['validation_rows_seen_in_train'],grouped['validation_rows_seen_in_train']],
       ['Mean length: training / validation',f"{naive['train_mean_length']:.1f} / {naive['validation_mean_length']:.1f}",f"{grouped['train_mean_length']:.1f} / {grouped['validation_mean_length']:.1f}"]],[240,135,135])
p('These are deterministic diagnostic simulations, not a saved or approved shared split. Row simulation: shuffle each class independently with Python Random(42 + label), then take round(20% of rows) for validation. Group simulation: shuffle exact-text groups per class with the same seed rule, then choose the prefix nearest to 20% of that class\'s rows. This is not sklearn train_test_split and need not reproduce the README\'s split.')
p(f"Additional sensitivity check: case-folding and collapsing whitespace gives {len(normalized):,} unique groups, {len(groups)-len(normalized)} fewer than exact matching, with {conflicts(normalized)} conflicting-label groups. This does not check semantic similarity or punctuation variants; choose and record the grouping policy before finalising the split.")

page('3. Recommended shared split protocol')
p('Recommended starting point: <b>80% training / 20% validation, seed 42, stratified by sentiment and grouped by review text</b>. Keep duplicate rows together to preserve the supplied class distribution. Do not silently drop duplicates: that changes sample weighting and should be a separately documented experiment.')
for title,text in [
('Define the grouping policy','Use exact text as the minimum protection. Prefer the conservative case-folded, whitespace-collapsed key if inspection confirms that merged variants are equivalent and no conflicting labels occur. Grouping normalization does not require altering model input text.'),
('Create a deterministic grouped split','Target approximately 216 negative and 1,264 positive validation rows (about 1,480 total). The original 7,401 rows do not divide exactly in an 80/20 ratio, and group sizes can shift counts slightly. Report realised counts rather than requiring an arbitrary exact total.'),
('Save the shared artifact','Export original zero-based row indices, seed, grouping rule, algorithm/version, source SHA-256 and realised counts. All group members must use this one artifact, including the TF-IDF baseline, GRU tuning and evaluation code.'),
('Validate before training','Assert index uniqueness, disjoint train/validation row indices, complete coverage of all rows, both labels present and zero overlap of grouping keys. Compare label ratios, length percentiles by class and long-review rates between partitions. Avoid repeatedly choosing a split based on model performance.'),
('Fit transforms on training only','Build TF-IDF vocabulary, IDF weights or neural vocabulary using the training partition. Derive loss/class weights from that partition. Tune hyperparameters and sequence length using validation; keep test data outside model-selection decisions.'),
('Plan fair evaluation','Report accuracy, macro-F1, per-class precision/recall/F1 and confusion matrix. Validation has only about 216 negative examples, so one missed negative changes negative recall by roughly 0.46 percentage points. Small score differences need cautious interpretation.')]:
    p(title,'Heading2');p(text)
p('Scope and reproducibility','Heading2')
p('No classifier was trained and no official split was written. Run .venv/bin/python analysis/train_split_report.py to regenerate this PDF, chart and train_split_statistics.json. The JSON records the source-file fingerprint and exact simulation results. Descriptive train-only evidence is sufficient for this preparation; product/reviewer identities and time-based leakage cannot be assessed because those identifiers are absent.')

def footer(c,d):
    c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526473'))
    c.drawString(42,24,'IE4483 | Train-only split preparation | 08 Nov 2026')
    c.drawRightString(553,24,str(d.page))
SimpleDocTemplate(str(OUT/'train_split_analysis.pdf'),pagesize=(595.28,841.89),leftMargin=42,rightMargin=42,topMargin=36,bottomMargin=46,title='IE4483 Training Set and Split Analysis').build(story,onFirstPage=footer,onLaterPages=footer)
print(json.dumps(stats,indent=2))
