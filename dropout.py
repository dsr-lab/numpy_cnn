import numpy as np


def cnn_dropout(weights, keep_probability):

    probabilities = np.random.rand(weights.shape[2], weights.shape[3]) < keep_probability
    new_weights = np.multiply(weights, probabilities)
    new_weights /= keep_probability

    return new_weights


def dense_dropout(weights, keep_probability):

    weights_shape = weights.shape

    probabilities = np.random.rand(weights.shape[0], weights.shape[1]) < keep_probability
    new_weights = np.multiply(weights, probabilities)
    new_weights /= keep_probability

    return new_weights


