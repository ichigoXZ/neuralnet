import numpy as np
from activations import sigmoid

def computeLoss (X, y, theta, options):
    J = 0
    if options["activations"] == "liner":
        J = np.sum(map(lambda x: x * x, (np.dot(X, theta) - y))) / (2 * y.shape[0])
    if options["activations"] == "sigmoid":
        h = sigmoid(np.dot(X, theta))
        J = -1 * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / y.shape[0]
    return J