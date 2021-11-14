import numpy as np


def softmax(scores):
    """
    Compute the softmax

    Parameters
    ----------
    scores : ndarray
        The values obtained from the last fully connected layer.
        The shape expected is (N, C), where N is the number of samples,
        and C is the number of classes.

    Returns
    -------
    result : ndarray
        The computed softmax
    """
    # the subtraction with np.max(scores) is required for having numerical stability
    e_x2 = np.exp(scores - np.max(scores))
    result = e_x2 / e_x2.sum(axis=1, keepdims=1)

    return result
