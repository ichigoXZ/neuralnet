import numpy as np
import dataGenerate as data
import itertools
import traceback
from matplotlib import pyplot as plt
from plot import plotSortScatter
import re

class Activation:
    class sigmoid:
        @staticmethod
        def output(Z):
            g = 1. / (1 + np.e ** (-Z))
            return g
        @staticmethod
        def der(Z):
            g = 1. / (1 + np.e ** (-Z))
            der = g * (1 - g)
            return der

    class tanh:
        @staticmethod
        def output(Z):
            g = np.tanh(Z)
            return g
        @staticmethod
        def der(Z):
            output = np.tanh(Z)
            return 1 - output ** 2


class Regularization:
    @staticmethod
    def L1(w):
        return np.abs(w)

    @staticmethod
    def L2(w):
        return 0.5 * w * w

class Errors:
    class SQUARE:
        @staticmethod
        def error(output, target):
            error = 0.5 * (output - target) ** 2
            return error
        @staticmethod
        def der(output, target):
            der = output - target
            return der

class Node:
    id = ""
    inputLinks = []
    outputs = []
    bias = 0.1
    totalInput = 0
    output = 0
    # Error derivative with respect to this node's output. */
    outputDer = 0
    # Error derivative with respect to this node's total input. */
    inputDer = 0
    # Accumulated error derivative with respect to this node's total input since
    # the last update. This derivative equals dE/db where b is the node's
    # bias term.
    accInputDer = 0
    # Number of accumulated err. derivatives with respect to the total input
    # since the last update.
    numAccumulatedDers = 0
    # Activation function that takes total input and returns node's output
    activation = Activation.sigmoid

    # Creates a new node with the provided id and activation function.
    def __init__(self, id, activation, initZero=False):
        self.id = id
        self.activation = activation
        self.inputLinks = []
        self.outputs = []
        if initZero:
            self.bias = 0

    # Recomputes the node's output and returns it.
    def updateOutput(self):
        # Stores total input into the node
        self.totalInput = self.bias
        for i in range(len(self.inputLinks)):
            link = self.inputLinks[i]
            self.totalInput += link.weight * link.source.output
        self.output = self.activation.output(self.totalInput)
        return self.output


class Link:
    id = ""
    source = Node
    destination = Node
    weight = np.random.random() - 0.5
    # Error derivative with respect to this weight.
    errorDer = 0;
    # Accumulated error derivative since the last update.
    accErrorDer = 0;
    # Number of accumulated derivatives since the last update.
    numAccumulatedDers = 0;

    regularization = Regularization.L1

    def __init__(self, source, destination, regularization, initZero=False):
        self.id = source.id + "-" + destination.id
        self.source = source
        self.destination = source
        self.regularization = regularization
        self.weight = np.random.random() - 0.5
        if initZero:
            self.weight = 0


def buildNetwork(
        networkShape, activation,
        outputActivation, regularization,
        inputIds, initZero=False):
    numLayers = len(networkShape)
    id = 1
    network = []
    # List of layers, with each layer being a list of nodes.
    for layerIdx in range(numLayers):
        isOutputLayer = (layerIdx == numLayers)
        isInputLayer = (layerIdx == 0)
        currentLayer = []

        numNodes = networkShape[layerIdx]
        for i in range(numNodes):
            nodeId = str(id)
            if isInputLayer:
                nodeId = inputIds[i]
            else:
                id += 1
            node = Node(nodeId, outputActivation
                    if isOutputLayer else activation, initZero)
            if layerIdx >= 1:
                # Add links from nodes in the previous layer to this node.
                for j in range(len(network[layerIdx - 1])):
                    prevNode = network[layerIdx - 1][j]
                    link = Link(prevNode, node, regularization, initZero)
                    node.inputLinks.append(link)
            currentLayer.append(node)
        network.append(currentLayer)

    return network


def forwardProp(network, inputs):
    """
    Runs a forward propagation of the provided input through the provided
    network. This method modifies the internal state of the network - the
    total input and output of each node in the network.
    :param network: The neural network.
    :param inputs:  THe input array. Its length should match the number of
        input nodes in the network.
    :return: The final output of the network.
    """
    inputLayer = network[0]
    if len(inputs) != len(inputLayer):
        raise Exception("The number of inputs must match the number of nodes in"
                        + "the input layer")
    # update the input layer
    for num, node in zip(inputs,inputLayer):
        node.output = num
    for layerIdx in range(1, len(network)):
        currentLayer = network[layerIdx]
        # Update all the nodes in this layer.
        for i in range(len(currentLayer)):
            node = currentLayer[i]
            node.updateOutput()
    return network[len(network) - 1][0].output

def backProp(network, target, errorFunc):
    """
    Runs a backward propagation using te provided target and the
    computed output of the previous call to forward propagation.
    :param network:
    :param target:
    """
    outputNode = network[len(network) - 1][0]
    outputNode.outputDer = errorFunc.der(outputNode.output, target)

    # Go through the layers backwards.
    for layerIdx in range(len(network) - 1, 0, -1):
        currentLayer = network[layerIdx]
        # Compute the error derivative of each node with respect to
        # 1) its total input
        # 2) each of its input weights.
        for i in range(len(currentLayer)):
            node = currentLayer[i]
            currentLayer[i].inputDer = node.outputDer * node.activation.der(node.totalInput)
            currentLayer[i].accInputDer += node.inputDer

            currentLayer[i].numAccumulatedDers += 1

        # Error derivative with respect to each weight coming into the node.
        for i in range(len(currentLayer)):
            node = currentLayer[i]
            for j in range(len(node.inputLinks)):
                link = node.inputLinks[j]
                currentLayer[i].inputLinks[j].errorDer = node.inputDer * link.source.output
                currentLayer[i].inputLinks[j].accErrorDer += link.errorDer
                currentLayer[i].inputLinks[j].numAccumulatedDers += 1
        if layerIdx == 1:
            continue
        prevLayer = net[layerIdx - 1]
        for i in range(len(prevLayer)):
            node = prevLayer[i]
            # Compute the error derivative with respect to each node'a output
            prevLayer[i].outputDer = 0
            for j in range(len(node.outputs)):
                output = node.outputs[j]
                prevLayer[i].outputDer += output.weight * output.destination.inputDer

# Update the weights of the network using the previously accumulated error derivatives.
def updateWeights(network, learningRate, regularizationRate):
    for layerIdx in range(1, len(network)):
        currentLayer = network[layerIdx]
        for i in range(len(currentLayer)):
            node = currentLayer[i]
            # Update the node's bias
            if node.numAccumulatedDers > 0:
                currentLayer[i].bias -= learningRate * node.accInputDer / node.numAccumulatedDers
                currentLayer[i].accInputDer = 0
                currentLayer[i].numAccumulatedDers = 0
            # Update the weights coming into this node.
            for j in range(len(node.inputLinks)):
                link = node.inputLinks[j]
                if link.regularization:
                    regularDer = link.regularization.der(link.weight)
                else:
                    regularDer = 0
                if link.numAccumulatedDers > 0:
                    currentLayer[i].inputLinks[j].weight -= (learningRate / link.numAccumulatedDers) * \
                                   (link.accErrorDer + regularizationRate * regularDer)
                    currentLayer[i].inputLinks[j].accErrorDer = 0
                    currentLayer[i].inputLinks[j].numAccumulatedDers = 0


# Returns the output node in the network
def getOutputNode(network):
    return network[len(network)-1][0]

def getLoss(network, dataPoints):
    loss = 0
    for i in range(len(dataPoints)):
        dataPoint = dataPoints[i]
        output = forwardProp(network, dataPoint[:-1])
        loss += Errors.SQUARE.error(output, dataPoint[-1])
    return loss / len(dataPoints)

def oneStep(iteration):
    iteration += 1

def predict(network):
    x_s = np.linspace(-5.0, 5.0, 100)
    y_s = np.linspace(-5.0, 5.0, 100)
    X, Y = np.meshgrid(x_s, y_s)
    X_s = list(itertools.chain.from_iterable(X))
    Y_s = list(itertools.chain.from_iterable(Y))
    inputs = np.c_[X_s, Y_s]
    Z_s = []
    for num,i in enumerate(inputs):
        output = forwardProp(network, i)
        Z_s.append(output)
    Z = np.c_[X_s, Y_s, Z_s]
    Z = [i for i in Z if i[2] > 0]
    Z = np.array(Z)
    print type(Z)
    plt.plot(Z[:, 0], Z[:, 1], 'gs')


if __name__ == "__main__":
    trainData = data.classifyXORData(100, 0)
    testData = data.classifyXORData(100, 0)
    iteration = 0
    alpha = 0.03
    net = buildNetwork([2,3,2,1], activation=Activation.tanh,
                       outputActivation=Activation.tanh, regularization=None,
                       inputIds=["x1","x2"])
    # for i in range(len(net)):
        # for j in range(len(net[i])):
        #     node = net[i][j]
        #     print i,"-",j,node.id
        #     print "link length:",len(node.inputLinks)
        #     for link in node.inputLinks:
        #         print link.source,"-",link.destination,"weight:",link.weight


    for iter in range(400):
        for i,point in enumerate(trainData):
            forwardProp(network=net, inputs=point[:-1])
            backProp(network=net, target=point[-1], errorFunc=Errors.SQUARE)
            if (i + 1) % 10 == 0:
                updateWeights(network=net, learningRate=alpha, regularizationRate=0)
        print getLoss(network=net, dataPoints=testData)
        print getLoss(network=net, dataPoints=trainData)

    predict(network=net)
    plotSortScatter(trainData)
    plt.show()

