"""Small reproducible validation grid for report question (c); never uses test."""
import argparse
import csv
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from train import ROOT, build_model, evaluate, load_shared_split


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results/baseline/tuning')
    parser.add_argument('--artifact-dir',type=Path,default=ROOT/'artifacts/tfidf_logreg_tuning')
    args=parser.parse_args()
    out=args.output_dir
    artifact=args.artifact_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit('Tuning results already exist; preserve the recorded run.')
    if (artifact/'best_model.joblib').exists():
        raise SystemExit('Best-model artifact exists; use a new --artifact-dir.')
    train,valid,manifest=load_shared_split(ROOT/'data')
    xt=[r['reviews'] for r in train]
    yt=np.array([r['sentiments'] for r in train])
    xv=[r['reviews'] for r in valid]
    yv=np.array([r['sentiments'] for r in valid])
    trials=[]
    best_model=None
    # One numerical thread for consistent within-grid comparisons.
    with threadpool_limits(limits=1):
        for ngram in (1,2):
            for weight in (None,'balanced'):
                for c in (.5,1.,4.):
                    model=build_model(c=c,ngram_max=ngram,class_weight=weight)
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning)
                        start=time.perf_counter();model.fit(xt,yt);elapsed=time.perf_counter()-start
                    prediction=model.predict(xv)
                    scores=evaluate(yv,prediction)
                    row={'ngram_max':ngram,'class_weight':'none' if weight is None else weight,'C':c,
                         'accuracy':scores['accuracy'],'macro_f1':scores['macro_f1'],
                         'negative_recall':scores['classification_report']['negative']['recall'],
                         'negative_precision':scores['classification_report']['negative']['precision'],
                         'feature_count':len(model.named_steps['tfidf'].vocabulary_),
                         'iterations':int(model.named_steps['classifier'].n_iter_[0]),
                         'fit_seconds_one_numerical_thread':elapsed,
                         'confusion_matrix':scores['confusion_matrix']}
                    trials.append(row)
                    # Highest macro-F1; exact ties retain the first configuration.
                    if best_model is None or row['macro_f1']>best['macro_f1']:
                        best=row;best_model=model
                    print(json.dumps(row),flush=True)
    out.mkdir(parents=True)
    evidence={'selection_metric':'validation macro-F1; exact ties retain first enumerated configuration',
              'grid':{'ngram_max':[1,2],'class_weight':['none','balanced'],'C':[.5,1.,4.]},
              'fixed_settings':{'min_df':2,'sublinear_tf':True,'smooth_idf':True,'norm':'l2',
                                'solver':'lbfgs','max_iter':2000,'tol':best_model.named_steps['classifier'].tol,
                                'random_state':42,'threshold':.5,'numerical_threads':1},
              'split_manifest_sha256':hashlib.sha256((ROOT/'data/split_manifest.json').read_bytes()).hexdigest(),
              'source_sha256':manifest['source_sha256'],'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__},
              'trials':trials,'best':best,
              'limits':'Single shared validation split; selected score is subject to model-selection optimism. Times are not directly comparable with the earlier baseline run, whose thread allocation was not recorded.'}
    (out/'tuning_results.json').write_text(json.dumps(evidence,indent=2)+'\n')
    fields=[k for k in trials[0] if k!='confusion_matrix']
    with (out/'tuning_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in trials)
    artifact.mkdir(parents=True,exist_ok=True)
    model_path=artifact/'best_model.joblib'
    if model_path.exists():
        raise FileExistsError(model_path)
    joblib.dump(best_model,model_path)
    print('Selected:',json.dumps(best))


if __name__=='__main__':
    main()
