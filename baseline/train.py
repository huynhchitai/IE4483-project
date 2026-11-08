"""TF-IDF + Logistic Regression on the saved shared split.

Run `python baseline/train.py --dry-run` to validate inputs without training.
Run `python baseline/train.py` to train and export evaluation artifacts.
"""
import argparse
import csv
import hashlib
import json
import platform
import resource
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
TOKEN_PATTERN = r"(?u)\b\w+(?:['’]\w+)*\b"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_shared_split(data_dir):
    manifest = json.loads((data_dir / 'split_manifest.json').read_text())
    source_path = data_dir / manifest['source']
    if sha256(source_path) != manifest['source_sha256']:
        raise ValueError('Source fingerprint differs from the shared split manifest')
    source = json.loads(source_path.read_text())
    parts, indices = [], []
    for name, key in [('train_splitted', 'train_indices'), ('valid_splitted', 'validation_indices')]:
        path = data_dir / f'{name}.json'
        if sha256(path) != manifest['output_sha256'][path.name]:
            raise ValueError(f'{path.name}: fingerprint mismatch')
        rows = json.loads(path.read_text())
        ids = manifest[key]
        if len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 or i >= len(source) for i in ids):
            raise ValueError(f'{name}: invalid or repeated source indices')
        if rows != [source[i] for i in ids]:
            raise ValueError(f'{name}: records differ from source indices')
        if not all(isinstance(r['reviews'], str) and r['reviews'].strip()
                   and type(r['sentiments']) is int and r['sentiments'] in (0, 1) for r in rows):
            raise ValueError(f'{name}: invalid review or label')
        if {r['sentiments'] for r in rows} != {0, 1}:
            raise ValueError(f'{name}: both classes are required')
        parts.append(rows)
        indices.append(ids)
    if set(indices[0]) & set(indices[1]) or set(indices[0]) | set(indices[1]) != set(range(len(source))):
        raise ValueError('Partitions must be disjoint and cover the source')
    if {r['reviews'] for r in parts[0]} & {r['reviews'] for r in parts[1]}:
        raise ValueError('Exact-text leakage between training and validation')
    return parts[0], parts[1], manifest


def build_model(*, c=4.0, ngram_max=2, min_df=2, class_weight='balanced'):
    return Pipeline([
        ('tfidf', TfidfVectorizer(lowercase=True, token_pattern=TOKEN_PATTERN,
                                stop_words=None, ngram_range=(1, ngram_max),
                                min_df=min_df, sublinear_tf=True, smooth_idf=True,
                                norm='l2', dtype=np.float64)),
        ('classifier', LogisticRegression(C=c, class_weight=class_weight,
                                         solver='lbfgs', max_iter=2000,
                                         random_state=42)),
    ])


def evaluate(y_true, y_pred):
    return {'accuracy': float(accuracy_score(y_true, y_pred)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro')),
            'classification_report': classification_report(
                y_true, y_pred, labels=[0, 1], target_names=['negative', 'positive'],
                output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            'confusion_matrix_order': 'rows=true, columns=predicted; labels=[0, 1]'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts' / 'tfidf_logreg')
    parser.add_argument('--dry-run', action='store_true', help='Validate files and show configuration; no fit or output files')
    parser.add_argument('--C', dest='c', type=float, default=4.0)
    parser.add_argument('--ngram-max', type=int, choices=[1, 2], default=2)
    parser.add_argument('--min-df', type=int, default=2)
    parser.add_argument('--class-weight', choices=['balanced', 'none'], default='balanced')
    args = parser.parse_args()
    if not np.isfinite(args.c) or args.c <= 0 or args.min_df < 1:
        parser.error('C must be positive and finite; min-df must be >= 1')
    train, valid, manifest = load_shared_split(args.data_dir)
    config = {'C': args.c, 'ngram_range': [1, args.ngram_max], 'min_df': args.min_df,
              'class_weight': None if args.class_weight == 'none' else 'balanced',
              'sublinear_tf': True, 'smooth_idf': True, 'norm': 'l2',
              'lowercase': True, 'stop_words': None, 'token_pattern': TOKEN_PATTERN,
              'dtype': 'float64', 'solver': 'lbfgs', 'regularization': 'L2 (solver default)',
              'max_iter': 2000, 'random_state': 42, 'decision_threshold': 0.5}
    counts = {name: {'rows': len(rows), 'labels': dict(Counter(r['sentiments'] for r in rows))}
              for name, rows in [('train', train), ('validation', valid)]}
    print(json.dumps({'config': config, 'data': counts, 'dry_run': args.dry_run}, indent=2))
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory is not empty; use a new --output-dir for each experiment')
    model = build_model(c=args.c, ngram_max=args.ngram_max, min_df=args.min_df,
                        class_weight=config['class_weight'])
    x_train = [r['reviews'] for r in train]
    y_train = np.array([r['sentiments'] for r in train])
    x_valid = [r['reviews'] for r in valid]
    y_valid = np.array([r['sentiments'] for r in valid])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    # One validation pass: predict_proba includes TF-IDF transform.
    probs = model.predict_proba(x_valid)[:, list(model.classes_).index(1)]
    pred = (probs > 0.5).astype(int)
    predict_seconds = time.perf_counter() - start
    result = evaluate(y_valid, pred)
    result['always_positive'] = evaluate(y_valid, np.ones(len(valid), dtype=int))
    features = {}
    for name, reviews in [('train', x_train), ('validation', x_valid)]:
        mat = model.named_steps['tfidf'].transform(reviews)
        features[name] = {'shape': list(mat.shape), 'nnz': int(mat.nnz),
                          'density': float(mat.nnz / (mat.shape[0] * mat.shape[1])),
                          'csr_array_bytes': int(mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes)}
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == 'darwin' else peak * 1024)
    result.update({'config': config, 'counts': counts, 'features': features,
                   'fit_seconds_including_vectorization': fit_seconds,
                   'validation_predict_proba_seconds_including_transform': predict_seconds,
                   'process_peak_rss_bytes': peak_bytes,
                   'memory_note': 'Process lifetime peak includes imports and diagnostics; CSR bytes count numeric arrays only, not vocabulary or Python objects.',
                   'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                                'numpy': np.__version__, 'joblib': joblib.__version__},
                   'split_source_sha256': manifest['source_sha256'],
                   'split_manifest_sha256': sha256(args.data_dir / 'split_manifest.json'),
                   'split_output_sha256': manifest['output_sha256'],
                   'classifier_iterations': model.named_steps['classifier'].n_iter_.tolist()})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / 'model.joblib'
    joblib.dump(model, model_path)
    result['model_file_bytes'] = model_path.stat().st_size
    with (args.output_dir / 'validation_predictions.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_index', 'reviews', 'sentiments', 'predictions', 'positive_probability'])
        writer.writerows((idx, row['reviews'], row['sentiments'], int(label), float(prob))
                         for idx, row, label, prob in zip(manifest['validation_indices'], valid, pred, probs))
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'accuracy': result['accuracy'], 'macro_f1': result['macro_f1'],
                      'output_dir': str(args.output_dir)}, indent=2))


if __name__ == '__main__':
    main()
