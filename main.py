import numpy as np
from dealCsv import readCsv,writeCsv
from gradientDescent import gradientDescent
from plot import plotLine,plotLoss,plot3D,plotSort
from featureNormalize import featureNormalize
from activations import sigmoid

loadfile = "data/ex2data1.csv"
options = {"alpha": 0.001,
           "iterations": 200000,    # for ex2data1: try >150000
           "activations": "sigmoid"
           }

def simpleLinerRegression(data):
    X = np.c_[data[:, :-1], np.ones(shape=data.shape[0])]
    y = data[:, -1]
    theta = np.ones(shape=data.shape[1])

    theta, loss = gradientDescent(X, y, theta, options)
    print theta

    plotLine(data, theta)
    plotLoss(loss, options["iterations"])

def mulLinerRegression(data):
    X = data[:,:-1]
    X_norm, mu, sigma = featureNormalize(X)
    X_norm = np.c_[X_norm, np.ones(shape=data.shape[0])]
    y = data[:, -1]
    theta = np.zeros(shape=data.shape[1])

    theta, loss = gradientDescent(X_norm, y, theta, options)
    plotLoss(loss, options["iterations"])
    print theta, mu, sigma

    plot3D(data, theta, mu, sigma)

    # test
    x = [[1380,3],[1494,3],[1940,4]]
    x = np.c_[(x - mu) / sigma, np.ones(3)]
    print np.dot(x, theta)

def logicRegression(data):
    X = np.c_[data[:, :-1], np.ones(shape=data.shape[0])]
    y = data[:, -1]
    theta = np.zeros(shape=data.shape[1])

    theta, loss = gradientDescent(X, y, theta, options)
    print theta

    # test (use train data)
    predict = (np.round(sigmoid(np.dot(X,theta))) == y)
    print 1.0 * np.sum(predict == True) / len(y)

    plotLoss(loss, options["iterations"])
    plotSort(data, theta)

if __name__ == '__main__':
    data = readCsv(loadfile)

    logicRegression(data)

