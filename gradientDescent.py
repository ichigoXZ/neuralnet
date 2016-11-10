import numpy as np
from computeLoss import computeLoss
from activations import sigmoid

def gradientDescent (X, y, theta, options):
    loss = []
    alpha = options["alpha"]
    iterations = options["iterations"]
    for i in range(iterations):
        if options["activations"] == "linear":
            theta = theta - alpha / y.shape[0] * np.dot(np.transpose(X), (np.dot(X, theta) - y))
        if options["activations"] == "sigmoid":
            theta = theta - alpha / y.shape[0] * np.dot(np.transpose(X), (sigmoid(np.dot(X, theta)) - y))
        loss.append(computeLoss(X, y, theta, options))
        print loss[i]
    return theta,loss


