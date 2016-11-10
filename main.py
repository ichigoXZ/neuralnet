import numpy as np
from dealCsv import readCsv,writeCsv
from gradientDescent import gradientDescent
from plot import plotLine,plotLoss,plot3D,plotSortScatter,plotSortLine,plotShow
from featureNormalize import featureNormalize
from activations import sigmoid
from featuremap import mapFeature

"""
for ex2data2: alpha = 0.0001 and degree = 6
              predict reach 0.6525423728881 when iterations reach 200000 then not change.
"""

loadfile = "data/ex2data2.csv"
options = {"alpha": 0.0001,
           "iterations": 10000,    # for ex2data1: try >150000
           "activations": "sigmoid",
           "regularized": True,
           "lambda": 1
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

def logicRegressionLine(data):
    X = np.c_[data[:, :-1], np.ones(shape=data.shape[0])]
    y = data[:, -1]
    theta = np.zeros(shape=data.shape[1])

    theta, loss = gradientDescent(X, y, theta, options)
    print theta

    # test (use train data)
    predict = (np.round(sigmoid(np.dot(X,theta))) == y)
    print 1.0 * np.sum(predict == True) / len(y)

    plotLoss(loss, options["iterations"])
    plotSortScatter(data)
    plotSortLine(data, theta)
    plotShow()

def logicRegressionRegularized(data):
    # plotSortScatter(data)
    # plotShow()
    X = mapFeature(data[:,:-1], 6)
    y = data[:,-1]
    theta = np.zeros(shape=X.shape[1])

    # theta, loss = gradientDescent(X, y, theta, options)
    for i in range(100):
        theta, _ = gradientDescent(X, y, theta, options)
        # test
        predict = (np.round(sigmoid(np.dot(X, theta))) == y)
        print i * options["iterations"], 1.0 * np.sum(predict == True) / len(y)
    # print theta

    # test
    predict = (np.round(sigmoid(np.dot(X, theta))) == y)
    print 1.0 * np.sum(predict == True) / len(y)
    print 1.0 * np.sum(y==0) / len(y)

    # plotLoss(loss, options["iterations"])
    # plotShow()

if __name__ == '__main__':
    data = readCsv(loadfile)

    logicRegressionRegularized(data)

