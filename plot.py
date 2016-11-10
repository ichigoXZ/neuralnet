import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from featuremap import mapFeature
from activations import sigmoid

def plotLine (data ,theta):
    a = np.floor(np.min(data, axis=0))
    b = np.floor(np.max(data, axis=0)) + 1
    # plt.figure(figsize=(b-a))
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

def plotSortScatter (data):
    zeroset = [i.tolist()[:-1] for i in data if i[-1]==0]
    oneset = [i.tolist()[:-1] for i in data if i[-1]==1]
    plt.plot([i[0] for i in zeroset], [i[1] for i in zeroset], 'rs',
                [i[0] for i in oneset], [i[1] for i in oneset], 'b^')

def plotSortLine(data, theta):
    # draw line
    x = np.array([np.floor(np.min(data[:,0])), np.floor(np.max(data[:,0])) + 1])
    Y = - (theta[0] * x + theta[2]) / theta[1]
    plt.plot(x, Y)

def plotSortBlock(data, theta):
    a = np.floor(np.min(data, axis=0))
    b = np.floor(np.max(data, axis=0)) + 1
    x_s = np.linspace(a[0], b[0], 100)
    y_s = np.linspace(a[1], b[1], 100)
    X, Y = np.meshgrid(x_s, y_s)
    X_s = list(itertools.chain.from_iterable(X))
    Y_s = list(itertools.chain.from_iterable(Y))

    Z =  np.c_[X_s, Y_s, sigmoid(np.dot(mapFeature(np.c_[X_s, Y_s], 6),theta))]
    Z = [i for i in Z if np.round(i[2])==0]
    Z = np.array(Z)
    print type(Z)
    plt.plot(Z[:,0],Z[:,1],'go')

def plotShow():
    plt.show()

def plotLoss (loss, iterations):
    plt.plot(range(iterations), loss)
    plt.show()