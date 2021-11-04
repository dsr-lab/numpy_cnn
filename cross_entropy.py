import numpy as np
import torch.nn.functional as F
import torch

def cross_entropy(scores, targets, t):

    # TODO: could be useful to clip prediction values.
    scores = np.clip(scores, 1e-8, 1. - 1e-8)

    n_predictions = scores.shape[0]
    #ce = -np.sum(targets.T*np.log(scores.T+1e-8))/n_predictions
    loss = -np.sum(targets * np.log(scores + 1e-8)) / n_predictions

    # loss = 0.
    # n_batch, n_class = scores.shape
    # # print(n_class)
    # for x1, y1 in zip(scores, t):
    #     class_index = int(y1)
    #     loss = loss + np.log(np.exp(x1[class_index]) / (np.exp(x1).sum()))
    # loss = - loss / n_batch




    return loss
