import numpy as np
from dealCsv import readCsv,writeCsv
from gradientDescent import gradientDescent

loadfile = "ex1data1.csv"
alpha = 0.01
iterations = 1500

if __name__ == '__main__':
    data = readCsv(loadfile)

    X = np.c_[data[:,:-1],np.ones(shape=data.shape[0])]
    y = data[:, -1]
    theta = np.zeros(shape=data.shape[1])

    theta = gradientDescent(X, y, theta, alpha=alpha, iterations=iterations)


