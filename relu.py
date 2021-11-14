def ReLU(x):
    """
    Apply the rectified linear unit opertion on the input

    Parameters
    ----------
    x : ndarray
        Inputs that must pass through the ReLU

    Returns
    -------
    x : ndarray
        The result of the computed ReLU operation
    """
    return (x > 0) * x


def dReLU(x):
    """
    Apply the derivative of the rectified linear unit

    Parameters
    ----------
    x : ndarray
        Inputs that must pass through the ReLU

    Returns
    -------
    x : ndarray
        The result of the computed derivative of the ReLU operation
    """
    return (x > 0) * 1.0

