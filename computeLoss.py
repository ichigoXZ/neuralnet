import numpy as np
from activations import sigmoid

def computeLoss (X, y, theta, options):
    J = 0
    m = y.shape[0]
    if options["activations"] == "linear":
        J = np.sum(map(lambda x: x * x, (np.dot(X, theta) - y))) / (2 * m )
    if options["activations"] == "sigmoid":
        h = sigmoid(np.dot(X, theta))
        if options["regularized"] != True:
            J = -1 * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / m
        else:
            theta_1 = theta[1:]
            J = -1 * np.sum(y * np.log(h) + (1-y) * np.log(1 - h)) / m \
                + options["lambda"] / (2 * m) * np.dot(np.transpose(theta_1), theta_1)
    return J