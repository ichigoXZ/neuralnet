import numpy as np

def mapFeature(data, degree):
    X = np.ones(data.shape[0])
    X1 = data[:,0]
    X2 = data[:,1]
    for i in range(1,degree+1):
        for j in range(i+1):
            X = np.c_[X, X1**(i-j) * X2**j]
    return X
