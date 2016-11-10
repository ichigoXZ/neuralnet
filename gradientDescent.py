import numpy as np
from computeLoss import computeLoss
from activations import sigmoid

def gradientDescent (X, y, theta, options):
    loss = []
    alpha = options["alpha"]
    iterations = options["iterations"]
    m = y.shape[0]
    for i in range(iterations):
        if options["activations"] == "linear":
            theta = theta - alpha / m * np.dot(np.transpose(X), (np.dot(X, theta) - y))
        if options["activations"] == "sigmoid":
            h = sigmoid(np.dot(X, theta))
            if options["regularized"] != True:
                theta = theta - alpha / m * np.dot(np.transpose(X), (h - y))
            else:
                theta_l = theta[1:]
                grad = (np.dot(np.transpose(X),h - y)) / m + np.r_[0,options["lambda"] * theta_l]
                theta -= alpha * grad
        loss.append(computeLoss(X, y, theta, options))
        print "loss:",loss[i]
    return theta,loss


