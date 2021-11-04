import numpy as np
import torch.nn.functional as F
import torch

def cross_entropy(scores, targets, t):

    # TODO: could be useful to clip prediction values.
    #scores = np.clip(scores, 1e-8, 1. - 1e-8)

    n_predictions = scores.shape[1]
    ce = -np.sum(targets*np.log(scores+1e-8))/n_predictions
    # ce2 = -(np.sum(targets * np.log(scores+1e-8) + (1 - targets) * np.log(1 - scores +1e-8)))/n_predictions

    # loss1 = F.cross_entropy(torch.as_tensor(scores.T), torch.as_tensor(t).long())
    #
    #
    #
    # loss = 0.
    # n_class, n_batch  = scores.shape
    # # print(n_class)
    # for x1, y1 in zip(scores.T, t):
    #     class_index = int(y1)
    #     loss = loss + np.log(np.exp(x1[class_index]) / (np.exp(x1).sum()))
    # loss = - loss / n_batch

    #print(loss, loss1)

    return ce
