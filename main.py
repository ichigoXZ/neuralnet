import numpy as np
from dealCsv import readCsv,writeCsv
from gradientDescent import gradientDescent
from plot import plotLine,plotLoss,plot3D,plotSortScatter,plotSortLine,plotShow,plotSortBlock
from featureNormalize import featureNormalize
from activations import sigmoid
from featuremap import mapFeature

"""
for ex2data2: alpha = 0.0001 and degree = 6 and initial theta all zero and lambda = 1
              predict reach 0.6525423728881 when iterations reach 200000 then not change.
              we can initial theta all one to reach higher accurate, but they all came to the same end
              change lamda can reach better results.
"""

loadfile = "data/ex2data2.csv"
options = {"alpha": 0.01,
           "iterations": 50,    # for ex2data1: try >150000
           "activations": "sigmoid",
           "regularized": True,
           "lambda": 0
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
    X = mapFeature(data[:,:-1], 6)
    y = data[:,-1]
    theta = np.zeros(shape=X.shape[1])

    theta, loss = gradientDescent(X, y, theta, options)
    accurates = []
    # for i in range(25):
    #     theta, _ = gradientDescent(X, y, theta, options)
        # test
        # predict = (np.round(sigmoid(np.dot(X, theta))) == y)
        # accurate = 1.0 * np.sum(predict == True) / len(y)
        # accurates.append(accurate)
        # print i * options["iterations"], accurate
    # print theta

    # test
    # predict = (np.round(sigmoid(np.dot(X, theta))) == y)
    # print 1.0 * np.sum(predict == True) / len(y)
    # print 1.0 * np.sum(y==0) / len(y)

    # plotLoss(accurates, 50)
    plotLoss(loss, options["iterations"])
    plotSortBlock(data,theta)
    plotSortScatter(data)
    plotShow()

def oneVsAll(images, labels, K):
    images = np.c_[images, np.ones(images.shape[0])]
    all_theta = np.zeros(shape=(K, images.shape[1]))


    losses = []
    for i in range(K):
        all_theta, loss = gradientDescent(images, (labels==i), all_theta[i], options)
        losses.append(loss)
        plotLoss(loss, options["iterations"])

    #test
    # plotLosses(losses, options["iterations"], K)

if __name__ == '__main__':
    # data = readCsv(loadfile)
    # simpleLinerRegression(data)   # ex1data1.csv
    # mulLinerRegression(data)      # ex1data2.csv
    # logicRegressionLine(data)     # ex2data1.csv
    # logicRegressionRegularized(data)    #ex2data2.csv
    images = readCsv("data/test_images.csv")
    labels = readCsv("data/test_labels.csv")
    oneVsAll(images, labels, K=10)
