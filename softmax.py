import numpy as np


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)

    e_x = np.exp(scores - np.max(scores))
    return e_x / e_x.sum(axis=0)

