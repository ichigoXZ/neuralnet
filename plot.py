import numpy as np
from matplotlib import pyplot as plt
import itertools
from mpl_toolkits.mplot3d import Axes3D
from featureNormalize import featureNormalize

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

def plot3D (data, theta, mu, sigma):
    x, y, z = data[:,0], data[:,1], data[:,2]
    a = np.floor(np.min(data, axis=0))
    b = np.floor(np.max(data, axis=0)) + 1
    x_s = np.linspace(a[0], b[0], 100)
    y_s = np.linspace(a[1], b[1], 100)
    X, Y = np.meshgrid(x_s, y_s)
    X_s = list(itertools.chain.from_iterable(X))
    Y_s = list(itertools.chain.from_iterable(Y))

    Z = np.dot(np.c_[(np.c_[X_s, Y_s]-mu)/sigma, np.ones(10000)], theta)

    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z.reshape(100,100), rstride=1, cstride=1, cmap='rainbow')
    ax.scatter(x,y,z)
    ax.set_zlabel('Z')
    plt.show()

def plotLoss (loss, iterations):
    plt.plot(range(iterations), loss)
    # plt.show()