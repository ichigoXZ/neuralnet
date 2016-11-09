import numpy as np
from computeLoss import computeLoss

def gradientDescent (X, y, theta, alpha, iterations):
    loss = []
    for i in range(iterations):
        theta = theta - alpha / y.shape[0] * np.dot(np.transpose(X), (np.dot(X, theta) - y))
        loss.append(computeLoss(X, y, theta))
        print loss[i]
    return theta,loss


