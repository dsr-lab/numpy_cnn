import numpy as np


def flatten(x):
    a = x.shape
    flatten_result = np.zeros(((x.shape[1]*x.shape[2]*x.shape[3]), x.shape[0]))
    f = flatten_result.shape
    for idx in range(x.shape[0]):
        image = x[idx, :, :, :]
        image = image.flatten().reshape(-1,)
        a = image.shape
        print(image.shape)
        flatten_result[:, idx] = image

        # row = x[idx]
        # flatten_result[idx] = np.flatten(row)
    #print(flatten_result.shape)
    #print("DONE")
    return flatten_result
