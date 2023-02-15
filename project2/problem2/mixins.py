import numpy as np


def w_calculator(x, y, delta):
    x = np.array(x)
    xt = x.transpose()
    xtx = np.dot(xt, x)
    delta_i = delta * np.identity(np.shape(xtx)[0])
    xtx += delta_i
    xtx_inverse = np.linalg.pinv(xtx)
    xtx_inverse_xt = np.dot(xtx_inverse, xt)
    w = np.dot(xtx_inverse_xt, y)
    return w


def f(x, w):
    s = 0
    for i in range(len(x)):
        s += x[i] * w[i]
    return s


def error(x, y, w):
    e = 0
    for i in range(len(x)):
        e += (y[i] - f(x[i], w))

    return e

