import numpy as np


def cross_entropy(predictions, targets):
    epsilon = 1e-12

    # TODO: could be useful to clip prediction values.
    # predictions = np.clip(predictions, epsilon, 1. - epsilon)

    n_predictions = predictions.shape[1]
    ce = -np.sum(targets*np.log(predictions+epsilon))/n_predictions

    return ce
