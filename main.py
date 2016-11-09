import numpy as np
from dealCsv import readCsv,writeCsv
from gradientDescent import gradientDescent
from plot import plotLine,plotLoss,plot3D
from featureNormalize import featureNormalize

loadfile = "ex1data2.csv"
alpha = 0.001
iterations = 1500

def simpleLinerRegression(data):
    X = np.c_[data[:, :-1], np.ones(shape=data.shape[0])]
    y = data[:, -1]
    theta = np.zeros(shape=data.shape[1])

    theta, loss = gradientDescent(X, y, theta, alpha=alpha, iterations=iterations)
    print theta

    plotLine(data, theta)
    plotLoss(loss, iterations)

def mulLinerRegression(data):
    X = data[:,:-1]
    X_norm, mu, sigma = featureNormalize(X)
    X_norm = np.c_[X_norm, np.ones(shape=data.shape[0])]
    y = data[:, -1]
    theta = np.zeros(shape=data.shape[1])

    theta, loss = gradientDescent(X_norm, y, theta, alpha=alpha, iterations=iterations)
    plotLoss(loss, iterations)
    print theta, mu, sigma

    plot3D(data, theta, mu, sigma)

    # test
    x = [[1380,3],[1494,3],[1940,4]]
    x = np.c_[(x - mu) / sigma, np.ones(3)]
    print np.dot(x, theta)

if __name__ == '__main__':
    data = readCsv(loadfile)

    # simpleLinerRegression(data)
    mulLinerRegression(data)