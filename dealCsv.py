import numpy as np
import string

def readTxt (filename):
    f = open(filename, 'rb')
    matrix = []
    for line in f:
        matrix.append([string.atof(item) for item in line.split(',')])
    return matrix

def readCsv (filename):
    matrix = np.loadtxt(open(filename, "rb"), delimiter=",", skiprows=0)
    return matrix

def writeCsv (filename, data):
    np.savetxt(filename, data, delimiter = ',')


