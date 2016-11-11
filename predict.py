import numpy as np
from plot import plotLoss
from activations import sigmoid
import itertools

def predictOneVsAll(images, labels, all_theta):
    images = np.c_[images, np.ones(images.shape[0])]
    rate = sigmoid(np.dot(images, np.transpose(all_theta)))
    predict = np.where(rate == np.max(rate, axis=1))
    predict = list(itertools.chain.from_iterable(predict))
    return 1.0 * np.sum((predict == labels) == True) / np.sum(labels)