import numpy as np

def computeLoss (X, y, theta):
    J =  np.sum(map(lambda x: x * x, (np.dot(X, theta) - y))) / (2 * y.shape[0])
    return J