import numpy as np

def sigmoid (Z):
    g = 1. / (1 + np.e**(-Z))
    return g