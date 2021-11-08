import numpy as np


def softmax(scores):

    e_x2 = np.exp(scores - np.max(scores))
    result = e_x2 / e_x2.sum(axis=1, keepdims=1)

    return result
