import numpy as np


def flatten(x):

    # flatten_result = np.zeros(((x.shape[1]*x.shape[2]*x.shape[3]), x.shape[0]))
    #
    # for idx in range(x.shape[0]):
    #     image = x[idx, :, :, :]
    #     image = image.flatten().reshape(-1,)
    #     # a = image.shape
    #     # print(image.shape)
    #     flatten_result[:, idx] = image

    flatten_result = np.ravel(x).reshape(x.shape[0], -1)

    return flatten_result
