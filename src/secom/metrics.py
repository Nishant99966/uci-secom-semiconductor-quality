import numpy as np
from sklearn.metrics import f1_score


def best_f1_threshold(y_true, probabilities):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(y_true) != len(probabilities) or not len(y_true):
        raise ValueError("Nonempty arrays of equal length required")
    candidates = np.unique(np.r_[np.linspace(.01, .99, 99), probabilities])
    best_threshold, best_score = .5, -1.0
    for threshold in candidates:
        score = f1_score(y_true, probabilities >= threshold, zero_division=0)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score
