import numpy as np
from plot import plotSortScatter, plotShow

def classifyCircleData(numSamples, noise):
    examples2D = []
    radius = 5
    def getCircleLabel(point, center):
        return 1 if (distance(point, center) < (radius * 0.5)) else -1

    # Generate positive points inside the circle.
    for i in range(numSamples/2):
        r = randUniform(0, radius * 0.5)
        angle = randUniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noiseX = randUniform(-radius, radius) * noise
        noiseY = randUniform(-radius, radius) * noise
        label = getCircleLabel([x + noiseX, y + noiseY], [0, 0])
        examples2D.append([x + noiseX, y + noiseY, label])

    # Generate negative points outside the circle.
    for i in range(numSamples/2):
        r = randUniform(0.7 * radius, radius)
        angle = randUniform(0, 2 * np.pi)
        x = r * np.sin(angle)
        y = r * np.cos(angle)
        noiseX = randUniform(-radius, radius) * noise
        noiseY = randUniform(-radius, radius) * noise
        label = getCircleLabel([x + noiseX, y + noiseY], [0, 0])
        examples2D.append([x + noiseX, y + noiseY, label])

    return np.array(examples2D)


def classifyXORData(numSamples, noise):
    examples2D = []
    def getXORLabel(point):
        return 1 if point[0] * point[1] >= 0 else -1

    for i in range(numSamples):
        padding = 0.3
        radius = 5
        x = randUniform(-radius, radius)
        x += padding if x > 0 else -padding
        y = randUniform(-radius, radius)
        y += padding if y > 0 else -padding
        noiseX = randUniform(-radius, radius) * noise
        noiseY = randUniform(-radius, radius) * noise
        label = getXORLabel([x + noiseX, y + noiseY])
        examples2D.append([x + noiseX, y + noiseY, label])

    return np.array(examples2D)

def classifySpiralData(numSamples, noise):
    examples2D = []
    n = numSamples/2
    def genSpiral(deltaT, label):
        for i in range(n):
            r = 1.0 * i / n * 5
            t = 1.75 * i / n *2 * np.pi + deltaT
            print r, t
            x = r * np.sin(t) + randUniform(-1, 1) * noise
            y = r * np.cos(t) + randUniform(-1, 1) * noise
            examples2D.append([x, y, label])

    genSpiral(0, 1)
    genSpiral(np.pi, -1)
    return np.array(examples2D)

def randUniform(a, b):
    return np.random.random() * (b - a) + a

def distance(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return np.sqrt(dx**2 + dy**2)

