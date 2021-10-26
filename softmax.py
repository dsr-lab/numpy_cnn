import numpy as np


def test_softmax():
    scores = [[3, -2, -100], [3, -2, -100]]
    scores = [[1, 2, 3, 6],
              [2, 4, 5, 6],
              [3, 8, 7, 6]]
    scores = [[-1, 0, 3, 5]]

    result = softmax(scores)
    print(result)
    print('softmax computed')


def softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)

    #denominator = np.sum(np.e ** scores, axis=1)
    #denominator = np.expand_dims(denominator, axis=1)

    #softmax_result = np.power(np.e, scores)
    #softmax_result = np.divide(softmax_result, denominator)

    e_x = np.exp(scores - np.max(scores))
    return e_x / e_x.sum(axis=0)

    #return softmax_result
