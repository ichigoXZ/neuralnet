import numpy as np
from matplotlib import pyplot as plt

def plotLine (data ,theta):
    a = np.floor(np.min(data, axis=0))
    b = np.floor(np.max(data, axis=0)) + 1
    plt.figure(figsize=(b-a))
    plt.scatter(data[:,0], data[:,-1],s=25, alpha=0.4, marker='o')
    x = np.array([a[0]-2,b[0]+2])
    x_a = np.c_[x, np.ones(shape=2)]
    Y = np.dot(x_a,theta)
    plt.plot(x,Y)
    plt.show()

def plotLoss (loss, iterations):
    plt.plot(range(iterations), loss)
    plt.show()