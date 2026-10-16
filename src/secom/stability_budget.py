"""Prospective-style SECOM inspection-budget experiment.

For each expanding-window time fold:
- select stable sensors using *only earlier training observations*;
- choose model + inspection threshold using internal chronological calibration;
- fit on the earlier observations and score the later validation period.
No final 20% holdout records are used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (average_precision_score, precision_score, recall_score,
                             matthews_corrcoef, roc_auc_score, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from .data import load_secom


def stable_columns(x, max_missing=0.60, max_shift=0.60, min_columns=15):
    """Use earlier/later *training* halves only to determine stable features."""
    mid = len(x)//2
    a, b = x.iloc[:mid], x.iloc[mid:]
    coverage = 1-x.isna().mean()
    ma, mb = a.mean(), b.mean()
    pooled = np.sqrt((a.var().fillna(0)+b.var().fillna(0))/2)
    mean_shift = ((ma-mb).abs()/(pooled+1e-8)).fillna(np.inf)
    missing_shift = (a.isna().mean()-b.isna().mean()).abs()
    valid = (coverage >= 1-max_missing)&(pooled>1e-8)
    score = (mean_shift+missing_shift).where(valid, np.inf)
    selected = score[score<=max_shift].sort_values().index.tolist()
    if len(selected)<min_columns:
        selected = score.sort_values().head(min(min_columns,int(valid.sum()))).index.tolist()
    if not selected:
        raise ValueError("No eligible stable sensor features")
    return selected


def models():
    return {
       "stable_logistic":Pipeline([
           ("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
           ("variance",VarianceThreshold()),
           ("scale",StandardScaler()),
           ("model",LogisticRegression(class_weight="balanced",max_iter=1500,random_state=42))
       ]),
       "stable_random_forest":Pipeline([
           ("impute",SimpleImputer(strategy="median",keep_empty_features=True)),
           ("variance",VarianceThreshold()),
           ("model",RandomForestClassifier(n_estimators=160,min_samples_leaf=4,
                                          class_weight="balanced_subsample",
                                          random_state=42,n_jobs=2))
       ])
    }


def inspection_scores(y, prob, budget):
    n=len(prob)
    # Select exactly floor(budget*n) largest risk scores; no test-label optimization.
    k=max(1,int(np.floor(budget*n)))
    ranked=np.argsort(-prob,kind="stable")
    flag=np.zeros(n,dtype=int)
    flag[ranked[:k]]=1
    y=np.asarray(y,dtype=int)
    return dict(recall=float(recall_score(y,flag,zero_division=0)),
                precision=float(precision_score(y,flag,zero_division=0)),
                mcc=float(matthews_corrcoef(y,flag)),
                flagged=int(flag.sum()),failure_count=int(y.sum()),
                detected=int(((flag==1)&(y==1)).sum()),
                false_alarms=int(((flag==1)&(y==0)).sum()),
                average_precision=float(average_precision_score(y,prob)),
                roc_auc=float(roc_auc_score(y,prob)) if len(np.unique(y))==2 else None)


def run(data_dir="data/raw",output_dir="results_stability",budget=0.25):
    if not (0<budget<1):
        raise ValueError("Budget must be between zero and one")
    x,y,dates=load_secom(data_dir)
    chronological=np.argsort(dates.to_numpy(),kind="stable")
    initial=chronological[:int(len(x)*.8)]
    x=x.iloc[initial].reset_index(drop=True)
    y=y.iloc[initial].reset_index(drop=True)
    folds=TimeSeriesSplit(n_splits=4)
    outcome={"protocol":"Forward-time folds; sensor filtering within training fold only; "
                        "top-quartile scoring on validation with fixed inspection budget.",
             "inspection_budget":budget,"training_population":len(y),"models":{}}
    predictions=[]
    for name,base in models().items():
        evaluations=[]
        for fold,(tr,va) in enumerate(folds.split(x),1):
            selected=stable_columns(x.iloc[tr])
            model=clone(base)
            model.fit(x.iloc[tr][selected],y.iloc[tr])
            prob=model.predict_proba(x.iloc[va][selected])[:,1]
            metrics=inspection_scores(y.iloc[va],prob,budget)
            metrics.update({"fold":fold,"selected_features":len(selected),
                            "train_count":len(tr),"validation_count":len(va)})
            evaluations.append(metrics)
            for row_idx,p in zip(va,prob):
                predictions.append({"model":name,"fold":fold,"row":int(row_idx),
                                    "actual":int(y.iloc[row_idx]),"probability":float(p)})
        outcome["models"][name]={"folds":evaluations,
                                 "mean_recall":float(np.mean([f["recall"] for f in evaluations])),
                                 "mean_precision":float(np.mean([f["precision"] for f in evaluations])),
                                 "mean_average_precision":float(np.mean([f["average_precision"] for f in evaluations])),
                                 "total_detected":sum(f["detected"] for f in evaluations),
                                 "total_failures":sum(f["failure_count"] for f in evaluations),
                                 "total_false_alarms":sum(f["false_alarms"] for f in evaluations)}
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"stability_budget_metrics.json").write_text(json.dumps(outcome,indent=2))
    pd.DataFrame(predictions).to_csv(out/"stability_budget_predictions.csv",index=False)
    print(json.dumps(outcome,indent=2))
    return outcome


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="data/raw")
    p.add_argument("--output-dir",default="results_stability")
    p.add_argument("--budget",type=float,default=.25)
    a=p.parse_args()
    run(a.data_dir,a.output_dir,a.budget)
