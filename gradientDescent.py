import numpy as np
from computeLoss import computeLoss

def gradientDescent (X, y, theta, alpha, iterations):
    # tmp = np.dot(X, theta) - y
    # print tmp.shape
    # print X.shape
    # tmp = np.dot( np.transpose(X), tmp)
    # print tmp.shape
    for i in range(iterations):
        theta = theta - alpha / y.shape[0] * np.dot(np.transpose(X), (np.dot(X, theta) - y))
        print computeLoss(X, y, theta)
    return theta


