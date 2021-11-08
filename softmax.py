import numpy as np
import torch
import torch.nn.functional as F


def softmax2(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def softmax(scores):
    # scores = np.asarray(scores, dtype=np.float64)

    # np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    e_x2 = np.exp(scores - np.max(scores))
    result = e_x2 / e_x2.sum(axis=1, keepdims=1)

    # result2 = softmax2(scores)

    return result
