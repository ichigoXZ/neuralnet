import numpy as np
from dealCsv import readCsv,writeCsv
from gradientDescent import gradientDescent
from plot import plotLine,plotLoss
from featureNormalize import featureNormalize

loadfile = "ex1data2.csv"
alpha = 0.01
iterations = 500

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
    X = np.c_[featureNormalize(X), np.ones(shape=data.shape[0])]
    y = data[:, -1]
    theta = np.zeros(shape=data.shape[1])

    theta, loss = gradientDescent(X, y, theta, alpha=alpha, iterations=iterations)
    plotLoss(loss, iterations)
    print theta


if __name__ == '__main__':
    data = readCsv(loadfile)

    # simpleLinerRegression(data)
    mulLinerRegression(data)