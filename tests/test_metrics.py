import numpy as np
from secom.metrics import best_f1_threshold

def test_threshold():
    threshold, score = best_f1_threshold(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert .2 < threshold <= .8
    assert score == 1.0
