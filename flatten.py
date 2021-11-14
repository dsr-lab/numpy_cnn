def flatten(x):
    """
    Flat the values coming from a convolutional layer.
    After this reshaping the values can pass through a dense layer

    Parameters
    ----------
    x : ndarray
        The values arriving from a convolutional layer

    Returns
    -------
    x_flattened : ndarray
        The flattened version of the original array
    """
    x_flattened = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
    return x_flattened
