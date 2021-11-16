import numpy as np


def cross_entropy(scores, targets):
    """
    Compute the cross entropy

    Parameters
    ----------
    scores : ndarray
        Scores obtained after passing the last fully connected layer through the softmax
    targets : int, optional
        One hot encoding labels
    Returns
    -------
    ce : float
        The cross entropy loss value
    """
    # Clip values for numerical stability
    scores = np.clip(scores, 1e-8, 1. - 1e-8)

    # Number of samples
    n_predictions = scores.shape[0]

    # Cross entropy computation
    ce = -np.sum(targets*np.log(scores+1e-8))/n_predictions

    return ce
