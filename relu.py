import numpy as np

def ReLU(x):
    return (x > 0) * x


def dReLU(x):
    return (x > 0) * 1.0

# def dReLU(x, dout):
#     dx = np.array(dout, copy=True)
#     # dx = np.ones_like(dout)
#     dx[x <= 0] = 0
#
#     return dx
