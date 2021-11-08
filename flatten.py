def flatten(x):
    x_flattened = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    return x_flattened
