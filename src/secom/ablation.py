"""Paired ablation: identical folds/models/budget, with versus without stable-sensor filtering.

The original last 20% chronological holdout is excluded. Rankings are selected using
only available earlier fold training features. All 4 strategies are reported.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
from .data import load_secom
from .stability_budget import stable_columns, models, inspection_scores


def run(data_dir="data/raw", output_dir="results_ablation", budget=0.25):
    x, y, dates = load_secom(data_dir)
    indices = np.argsort(dates.to_numpy(), kind="stable")[:int(len(y)*0.8)]
    x = x.iloc[indices].reset_index(drop=True)
    y = y.iloc[indices].reset_index(drop=True)
    output = {"protocol":"Paired ablation on identical expanding-time validation folds",
              "budget":budget,"rows_available":len(y),"folds":4,"models":{}}
    predictions = []
    for name, estimator in models().items():
        variants = {}
        for filtered in (False, True):
            label = "stable_filtered" if filtered else "all_sensors"
            fold_metrics = []
            for fold, (train_ix, valid_ix) in enumerate(TimeSeriesSplit(n_splits=4).split(x), 1):
                chosen = stable_columns(x.iloc[train_ix]) if filtered else list(x.columns)
                model = clone(estimator)
                model.fit(x.iloc[train_ix][chosen], y.iloc[train_ix])
                probs = model.predict_proba(x.iloc[valid_ix][chosen])[:, 1]
                m = inspection_scores(y.iloc[valid_ix], probs, budget)
                m.update({"fold":fold,"features":len(chosen),"train_rows":len(train_ix),
                          "valid_rows":len(valid_ix)})
                fold_metrics.append(m)
                for row, score in zip(valid_ix, probs):
                    predictions.append({"model":name, "variant":label, "fold":fold,
                                        "chronological_training_index":int(row),
                                        "actual":int(y.iloc[row]),"score":float(score)})
            variants[label] = {
                "folds":fold_metrics,
                "true_defects":int(sum(m["failure_count"] for m in fold_metrics)),
                "detected":int(sum(m["detected"] for m in fold_metrics)),
                "false_alarms":int(sum(m["false_alarms"] for m in fold_metrics)),
                "flagged":int(sum(m["flagged"] for m in fold_metrics)),
                "mean_fold_ap":float(np.mean([m["average_precision"] for m in fold_metrics])),
                "pooled_recall":float(sum(m["detected"] for m in fold_metrics) /
                                      max(1,sum(m["failure_count"] for m in fold_metrics))),
            }
        output["models"][name]=variants
        output.setdefault("paired_deltas",{})[name] = {
            "detected_stable_minus_all":variants["stable_filtered"]["detected"]-variants["all_sensors"]["detected"],
            "false_alarms_stable_minus_all":variants["stable_filtered"]["false_alarms"]-variants["all_sensors"]["false_alarms"],
            "mean_ap_stable_minus_all":variants["stable_filtered"]["mean_fold_ap"]-variants["all_sensors"]["mean_fold_ap"],
        }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out/"ablation_metrics.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    pd.DataFrame(predictions).to_csv(out/"ablation_predictions.csv",index=False)
    print(json.dumps(output,indent=2))
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="data/raw")
    parser.add_argument("--output-dir",default="results_ablation")
    parser.add_argument("--budget",type=float,default=.25)
    args=parser.parse_args()
    run(args.data_dir,args.output_dir,args.budget)
