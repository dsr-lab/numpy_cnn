import numpy as np


def flatten(x):
    # flatten_result = np.zeros(((x.shape[1]*x.shape[2]*x.shape[3]), x.shape[0]))
    # f = flatten_result.shape
    # for idx in range(x.shape[0]):
    #     image = x[idx, :, :, :]
    #     image = image.flatten().reshape(-1,)
    #
    #     flatten_result[:, idx] = image

    a = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    b = a.T
    return b
