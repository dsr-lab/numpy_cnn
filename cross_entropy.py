import numpy as np


def cross_entropy(scores, targets):
    # Clip values for numeric stability
    scores = np.clip(scores, 1e-8, 1. - 1e-8)

    # Number of samples
    n_predictions = scores.shape[0]

    # Cross entropy computation
    ce = -np.sum(targets*np.log(scores+1e-8))/n_predictions

    return ce
