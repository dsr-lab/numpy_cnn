import numpy as np


def cross_entropy(scores, targets):

    # TODO: could be useful to clip prediction values.
    # predictions = np.clip(predictions, epsilon, 1. - epsilon)

    n_predictions = scores.shape[1]
    ce = -np.sum(targets*np.log(scores+1e-8))/n_predictions

    return ce
