"""Verify an existing baseline run and export its report to results/baseline/; never refit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak, Image

from train import ROOT, evaluate, load_shared_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'artifacts/tfidf_logreg')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results'/'baseline')
    args = parser.parse_args()
    metrics_path = args.run_dir/'metrics.json'
    m = json.loads(metrics_path.read_text())
    train, valid, manifest = load_shared_split(ROOT/'data')
    manifest_hash = hashlib.sha256((ROOT/'data/split_manifest.json').read_bytes()).hexdigest()
    if m['split_manifest_sha256'] != manifest_hash or m['split_output_sha256'] != manifest['output_sha256']:
        raise ValueError('Run does not match the shared split')
    model_path = args.run_dir/'model.joblib'
    model = joblib.load(model_path)
    reviews = [r['reviews'] for r in valid]
    truth = np.array([r['sentiments'] for r in valid])
    prob = model.predict_proba(reviews)[:, list(model.classes_).index(1)]
    pred = (prob > .5).astype(int)
    measured = evaluate(truth, pred)
    for key in ('accuracy','macro_f1'):
        if not np.isclose(measured[key],m[key],rtol=0,atol=1e-12):
            raise ValueError(f'{key} does not match saved metrics')
    if measured['confusion_matrix'] != m['confusion_matrix']:
        raise ValueError('Confusion matrix mismatch')
    with (args.run_dir/'validation_predictions.csv').open(newline='') as f:
        saved = list(csv.DictReader(f))
    if len(saved) != len(valid):
        raise ValueError('Prediction row count mismatch')
    for i, row in enumerate(saved):
        if (int(row['source_index']) != manifest['validation_indices'][i]
            or row['reviews'] != reviews[i] or int(row['sentiments']) != int(truth[i])
            or int(row['predictions']) != int(pred[i])
            or not np.isclose(float(row['positive_probability']),prob[i],rtol=0,atol=1e-12)):
            raise ValueError('Prediction row mismatch')
    for name, records in [('train',train),('validation',valid)]:
        x = model.named_steps['tfidf'].transform([r['reviews'] for r in records])
        feature = m['features'][name]
        if list(x.shape) != feature['shape'] or x.nnz != feature['nnz'] or x.data.nbytes+x.indices.nbytes+x.indptr.nbytes != feature['csr_array_bytes']:
            raise ValueError('Feature statistics mismatch')
    if model_path.stat().st_size != m['model_file_bytes']:
        raise ValueError('Serialized model size mismatch')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'baseline_metrics.json').write_bytes(metrics_path.read_bytes())
    verification = {
        'run_directory': str(args.run_dir.relative_to(ROOT)) if args.run_dir.is_relative_to(ROOT) else str(args.run_dir),
        'metrics_sha256': hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'checks': ['shared split hashes', 'saved model validation scores', 'prediction rows and probabilities',
                   'feature shapes, nonzeros and CSR array bytes', 'serialized model size'],
        'training_repeated': False,
        'timing_and_peak_memory': 'Copied from the original run; verification inference is not included.'}
    (args.output_dir/'baseline_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    mib = lambda b: b/2**20
    trainf, validf = m['features']['train'],m['features']['validation']
    perf = [['Model','Accuracy','Macro-F1','Negative recall'],
            ['TF-IDF + Logistic Regression',f"{m['accuracy']:.2%}",f"{m['macro_f1']:.4f}",f"{m['classification_report']['negative']['recall']:.2%}"],
            ['Always positive',f"{m['always_positive']['accuracy']:.2%}",f"{m['always_positive']['macro_f1']:.4f}",'0.00%']]
    classes = [['Class','Precision','Recall','F1','Support']]
    for name in ('negative','positive'):
        r=m['classification_report'][name]
        classes.append([name,*[f'{r[k]:.4f}' for k in ('precision','recall','f1-score')],int(r['support'])])
    resource_rows = [
        ['Measurement','Value'],
        ['Fit wall time (TF-IDF + classifier)',f"{m['fit_seconds_including_vectorization']:.4f} s"],
        ['Validation predict_proba wall time (including transform)',f"{m['validation_predict_proba_seconds_including_transform']:.4f} s / 1,480 reviews"],
        ['Average batch inference time per review',f"{1000*m['validation_predict_proba_seconds_including_transform']/1480:.4f} ms"],
        ['Process lifetime peak RSS',f"{mib(m['process_peak_rss_bytes']):.2f} MiB"],
        ['Training feature shape',str(trainf['shape'])],
        ['Validation feature shape',str(validf['shape'])],
        ['Training nonzeros / density',f"{trainf['nnz']:,} / {trainf['density']:.4%}"],
        ['Training CSR numeric arrays',f"{mib(trainf['csr_array_bytes']):.3f} MiB"],
        ['Validation CSR numeric arrays',f"{mib(validf['csr_array_bytes']):.3f} MiB"],
        ['Hypothetical dense training float64 array',f"{mib(np.prod(trainf['shape'])*8):,.2f} MiB (calculated, not allocated)"],
        ['Serialized vectorizer + classifier',f"{mib(m['model_file_bytes']):.3f} MiB"],
        ['Solver iterations',str(m['classifier_iterations'][0])]]
    def md_table(rows):
        return '\n'.join(['| '+' | '.join(map(str,rows[0]))+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows[1:]])
    intro = 'Evaluation uses the shared training partition (5,921 reviews) and validation partition (1,480 reviews: 216 negative, 1,264 positive), with seed 42 and no exact-text overlap. This report verifies and summarizes the existing artifacts/tfidf_logreg run; it does not retrain the model or use the official test set.'
    config = 'Word unigram + bigram TF-IDF; min_df=2; sublinear TF; smoothed IDF; L2-normalized float64 vectors; lowercase, contraction-preserving tokenizer, no stop-word removal. Logistic Regression: C=4, balanced class weights, lbfgs, max_iter=2000, random_state=42. Vocabulary and IDF are fitted on training only; decision threshold is 0.5 (positive when probability > 0.5).'
    interpretation = f"The baseline correctly classifies {int((pred==truth).sum()):,} of 1,480 validation reviews. Accuracy exceeds the always-positive reference by {(m['accuracy']-m['always_positive']['accuracy'])*100:.2f} percentage points. It detects {m['confusion_matrix'][0][0]} of 216 negative reviews, missing {m['confusion_matrix'][0][1]}, and incorrectly labels {m['confusion_matrix'][1][0]} positive reviews as negative. Minority-class F1 is lower than positive-class F1. These are single-split results for the initial configuration, not a completed tuning study or a final test score."
    limits = 'Fit time includes training vectorization and classifier fitting, but excludes data loading, validation, serialization and reporting. Validation time includes vectorization and probability prediction for one batch; per-review time is batch time divided by 1,480, not standalone request latency. Peak RSS is the process lifetime peak captured after feature diagnostics, including imports and Python overhead; it is not incremental model RAM or a separately isolated fit peak, and excludes later serialization. CSR sizes count data/indices/indptr arrays only, excluding vocabulary and object overhead. Serialized file size is disk storage. The hypothetical dense-array size is a calculation, not measured RAM usage.'
    comparison = 'Runtime and memory are observations from one saved run, not repeated benchmark averages. The original metrics do not record CPU model, allocated threads, BLAS configuration or host load; these cannot be reconstructed reliably after the run. For the future GRU comparison, record hardware, thread/device settings, timing boundaries, peak-memory method, versions and the same split. Compare both feature storage and end-to-end pipeline resources. Since representation and classifier both differ, do not attribute all accuracy differences to feature format alone. GRU results remain pending.'
    versions = ', '.join(f'{k} {v}' for k,v in m['versions'].items())
    text = '# TF-IDF + Logistic Regression: validation results\n\n'+intro+'\n\n## Configuration\n\n'+config+'\n\n## Predictive performance\n\n'+md_table(perf)+'\n\n'+md_table(classes)+'\n\n### Confusion matrix\n\nRows are true labels; columns are predicted labels.\n\n'+md_table([['True / predicted','Negative (0)','Positive (1)'],['Negative (0)',*m['confusion_matrix'][0]],['Positive (1)',*m['confusion_matrix'][1]]])+'\n\n![Validation confusion matrix](baseline_confusion_matrix.png)\n\n'+interpretation+'\n\n## Resource consumption\n\n'+md_table(resource_rows)+'\n\n'+limits+'\n\n## Future comparison for question (f)\n\n'+comparison+'\n\n## Reproducibility\n\nVersions: '+versions+'.\n\nOriginal measurements: [baseline_metrics.json](baseline_metrics.json). Verification: [baseline_verification.json](baseline_verification.json). Model and row-level predictions remain in artifacts/tfidf_logreg/ (ignored by Git).\n\n```bash\n# Re-export and verify this existing run without refitting\n.venv/bin/python baseline/report_results.py\n\n# To measure a new run, choose a fresh artifact directory\n.venv/bin/python baseline/train.py --output-dir artifacts/tfidf_logreg_new\n.venv/bin/python baseline/report_results.py --run-dir artifacts/tfidf_logreg_new --output-dir results/baseline/new_run\n```\n'
    (args.output_dir/'baseline_report.md').write_text(text)
    with (args.output_dir/'baseline_summary.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['model','accuracy','macro_f1','negative_recall','fit_seconds','validation_predict_seconds','peak_process_rss_bytes','training_csr_bytes','feature_dimensions','model_file_bytes'])
        writer.writerow(['TF-IDF + Logistic Regression',m['accuracy'],m['macro_f1'],m['classification_report']['negative']['recall'],m['fit_seconds_including_vectorization'],m['validation_predict_proba_seconds_including_transform'],m['process_peak_rss_bytes'],trainf['csr_array_bytes'],trainf['shape'][1],m['model_file_bytes']])
    fig,ax=plt.subplots(figsize=(4.5,3),layout='constrained')
    cm=np.array(m['confusion_matrix'])
    ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>600 else '#153951',fontsize=15)
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],xlabel='Predicted label',ylabel='True label',title='Shared validation set (n = 1,480)')
    fig.savefig(args.output_dir/'baseline_confusion_matrix.png',dpi=180)
    plt.close(fig)
    styles=getSampleStyleSheet()
    styles['BodyText'].fontSize=9.5;styles['BodyText'].leading=13
    styles['Heading1'].textColor=colors.HexColor('#153951')
    story=[]
    def p(s,style='BodyText'):
        story.extend([Paragraph(escape(s),styles[style]),Spacer(1,7)])
    def table(rows,widths):
        t=Table([[Paragraph(escape(str(v)),styles['BodyText']) for v in r] for r in rows],colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff5')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d4dee5'))]))
        story.extend([t,Spacer(1,9)])
    p('TF-IDF + Logistic Regression','Title');p('Validation performance | 8 November 2026','Heading2');p(intro);p('Configuration','Heading2');p(config)
    table(perf,[205,95,100,111]);table(classes,[115,100,100,100,96])
    story.append(Image(str(args.output_dir/'baseline_confusion_matrix.png'),width=288,height=192));p(interpretation)
    story.append(PageBreak());p('Resource consumption','Heading1');table(resource_rows,[320,191]);p(limits)
    story.append(PageBreak());p('Interpretation and comparison plan','Heading1');p(comparison);p('Reproducibility and evidence','Heading2');p('Versions: '+versions+'.');p('Training and validation fingerprints are preserved in baseline_metrics.json. Verification checks the saved model against every validation prediction and checks feature statistics against the original run. No model fitting is performed while generating this report.');p('Original model and predictions: artifacts/tfidf_logreg/. Exported metrics: results/baseline/baseline_metrics.json. Verification fingerprints: results/baseline/baseline_verification.json.');p('Re-export: .venv/bin/python baseline/report_results.py');p('Comparison table: results/baseline/baseline_summary.csv. GRU results will be stored separately under results/gru/ when available.');p('This report supplies the baseline evidence for question (f). It does not establish which feature representation or classifier is best.')
    def footer(c,d):
        c.setFont('Helvetica',8);c.drawString(42,24,'IE4483 | Baseline validation and resource measurements');c.drawRightString(553,24,str(d.page))
    SimpleDocTemplate(str(args.output_dir/'baseline_report.pdf'),pagesize=(595.28,841.89),leftMargin=42,rightMargin=42,topMargin=36,bottomMargin=45,title='IE4483 Baseline Results').build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'output_dir':str(args.output_dir),'verified':True},indent=2))


if __name__=='__main__':
    main()
