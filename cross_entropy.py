import numpy as np
import torch.nn.functional as F
import torch

def cross_entropy(scores, targets, t):

    # TODO: could be useful to clip prediction values.
    scores = np.clip(scores, 1e-8, 1. - 1e-8)

    n_predictions = scores.shape[1]
    ce = -np.sum(targets*np.log(scores+1e-8))/n_predictions



    loss = F.cross_entropy(torch.as_tensor(scores.T), torch.as_tensor(t.T).long())


    return ce
