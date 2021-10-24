def ReLU(x):
    return (x > 0) * x


def dReLU(x):
    return (x > 0) * 1.0
