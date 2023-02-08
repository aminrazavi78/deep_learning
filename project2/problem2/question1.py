import math

import numpy as np


def expect(x: list):
    return np.sum(x)/len(x)


def var(x):
    return expect((x - expect(x))**2)


def standard(x):
    return (x - expect(x))/math. sqrt(var(x))
