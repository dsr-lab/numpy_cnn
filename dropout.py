import numpy as np


def cnn_dropout(weights, keep_probability):
    """
    Compute the dropout operation to a convolutional layer

    Parameters
    ----------
    weights : ndarray
        The array that pass through the dropout operation.
        The expected shape is (N, C, H, W), where:
        - N = number of images
        - C = number of channels
        - H = height
        - W = width
    keep_probability : int
        Probability of not shutting down a neuron

    Returns
    -------
    new_weights : ndarray
        The original array with some neurons shut down
    """
    probabilities = np.random.rand(weights.shape[2], weights.shape[3]) < keep_probability
    new_weights = np.multiply(weights, probabilities)
    new_weights /= keep_probability

    return new_weights


def dense_dropout(weights, keep_probability):
    """
        Compute the dropout operation to a dense layer

        Parameters
        ----------
        weights : ndarray
            The array that pass through the dropout operation.
            The expected shape is (N, H), where:
            - N = number of images
            - H = number of hidden layers

        keep_probability : int
            Probability of not shutting down a neuron

        Returns
        -------
        new_weights : ndarray
            The original array with some neurons shut down
        """

    probabilities = np.random.rand(weights.shape[0], weights.shape[1]) < keep_probability
    new_weights = np.multiply(weights, probabilities)
    new_weights /= keep_probability

    return new_weights


