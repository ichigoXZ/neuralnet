import numpy as np
import dataGenerate as data
import itertools
import traceback
from matplotlib import pyplot as plt
from plot import plotSortScatter, plotLoss



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

    class Relu:
        @staticmethod
        def output(Z):
            return np.max(0, Z)
        @staticmethod
        def der(Z):
            j = 0 if Z<=0 else 1
            return j

    class linear:
        @staticmethod
        def output(Z):
            return Z
        @staticmethod
        def der(Z):
            return 1


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
        self.destination = destination
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
                    # prevNode = network[layerIdx - 1][j]
                    # print "[1]prevNode.id", network[layerIdx - 1][j].id," node.id", node.id
                    link = Link(network[layerIdx - 1][j], node, regularization, initZero)
                    network[layerIdx - 1][j].outputs.append(link)
                    # print "[2]link.id", network[layerIdx - 1][j].outputs[-1].id
                    # print "[2]link.source:", link.source.id, " link.dest:", link.destination.id
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
    for num, node in zip(inputs,network[0]):
        node.output = num
    for layerIdx in range(1, len(network)):
        currentLayer = network[layerIdx]
        # Update all the nodes in this layer.
        for i in range(len(currentLayer)):
            network[layerIdx][i].updateOutput()
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
    network[len(network) - 1][0].outputDer = outputNode.outputDer

    # Go through the layers backwards.
    for layerIdx in range(len(network) - 1, 0, -1):
        # print "[1]layerIdx:",layerIdx
        currentLayer = network[layerIdx]
        # Compute the error derivative of each node with respect to
        # 1) its total input
        # 2) each of its input weights.
        for i in range(len(currentLayer)):
            # print "  [2-bias]i:",i
            node = network[layerIdx][i]
            network[layerIdx][i].inputDer = node.outputDer * node.activation.der(node.totalInput)
            network[layerIdx][i].accInputDer +=  network[layerIdx][i].inputDer

            network[layerIdx][i].numAccumulatedDers += 1

        # Error derivative with respect to each weight coming into the node.
        currentLayer = network[layerIdx]
        for i in range(len(currentLayer)):
            # print "  [2-weight]i:",i
            node = network[layerIdx][i]
            for j in range(len(node.inputLinks)):
                # print "    [3]j",j
                link = node.inputLinks[j]
                network[layerIdx][i].inputLinks[j].errorDer = node.inputDer * link.source.output
                network[layerIdx][i].inputLinks[j].accErrorDer += network[layerIdx][i].inputLinks[j].errorDer
                # print "    [3]link.accErrorDer:", network[layerIdx][i].inputLinks[j].accErrorDer
                network[layerIdx][i].inputLinks[j].numAccumulatedDers += 1
        if layerIdx == 1:
            continue
        prevLayer = network[layerIdx - 1]
        for i in range(len(prevLayer)):
            # print "  [2-prev]i:",i
            node = network[layerIdx - 1][i]
            # Compute the error derivative with respect to each node'a output
            network[layerIdx - 1][i].outputDer = 0
            for j in range(len(node.outputs)):
                output = node.outputs[j]
                # print "    [3]output.weight:",output.weight
                # print "    [3]output.source.id:",output.source.id," output.dest.id:", output.destination.id
                network[layerIdx - 1][i].outputDer += output.weight * output.destination.inputDer
            # print "  [2]prevLayer.outputDer:", network[layerIdx - 1][i].outputDer

# Update the weights of the network using the previously accumulated error derivatives.
def updateWeights(network, learningRate, regularizationRate):
    for layerIdx in range(1, len(network)):
        currentLayer = network[layerIdx]
        for i in range(len(currentLayer)):
            node = network[layerIdx][i]
            # Update the node's bias
            if node.numAccumulatedDers > 0:
                network[layerIdx][i].bias -= learningRate * node.accInputDer / node.numAccumulatedDers
                network[layerIdx][i].accInputDer = 0
                network[layerIdx][i].numAccumulatedDers = 0
            # Update the weights coming into this node.
            node = network[layerIdx][i]
            for j in range(len(node.inputLinks)):
                link = node.inputLinks[j]
                if link.regularization:
                    regularDer = link.regularization.der(link.weight)
                else:
                    regularDer = 0
                if link.numAccumulatedDers > 0:
                    network[layerIdx][i].inputLinks[j].weight -= (learningRate / link.numAccumulatedDers) * \
                                   (link.accErrorDer + regularizationRate * regularDer)
                    network[layerIdx][i].inputLinks[j].accErrorDer = 0
                    network[layerIdx][i].inputLinks[j].numAccumulatedDers = 0


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

def predictPlot(network, data):
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
    plt.plot(Z[:, 0], Z[:, 1], 'gs')
    plotSortScatter(data)
    plt.show()

def predict(network, data):
    result = []
    for item in data:
        result.append(forwardProp(network=network, inputs=item))
    return result

def train(network, iteration, trainData, batchSize=10,
          errorFunc=Errors.SQUARE, learningRate=0.03, regularizationRate=0):
    for iter in range(iteration):
        for i, point in enumerate(trainData):
            forwardProp(network=network, inputs=point[:-1])
            backProp(network=net, target=point[-1],errorFunc=errorFunc)
            if (i + 1) % batchSize == 0:
                updateWeights(network=network, learningRate=learningRate, regularizationRate=regularizationRate)




def oneStep(network, iteration, trainData, batchSize=10,
          errorFunc=Errors.SQUARE, learningRate=0.03, regularizationRate=0):
    for i, point in enumerate(trainData):
        forwardProp(network=network, inputs=point[:-1])
        backProp(network=net, target=point[-1], errorFunc=errorFunc)
        if (i + 1) % batchSize == 0:
            updateWeights(network=network, learningRate=learningRate, regularizationRate=regularizationRate)

if __name__ == "__main__":
    trainData = data.classifyCircleData(100, 0)
    testData = data.classifyCircleData(100, 0.1)
    iteration = 400
    alpha = 0.03
    net = buildNetwork([2,3,2,1], activation=Activation.tanh,
                       outputActivation=Activation.tanh, regularization=None,
                       inputIds=["x1","x2"])

    # trainLoss, testLoss = train(network=net, iteration=iteration, trainData=trainData)
    trainLoss = []
    testLoss = []
    for i in range(iteration):
        oneStep(network=net, iteration=iteration, trainData=trainData)
        trainLoss.append(getLoss(network=net, dataPoints=trainData))
        testLoss.append(getLoss(network=net, dataPoints=testData))
        print "step:",i,"  loss:",getLoss(network=net, dataPoints=trainData)

    plotLoss(trainLoss, iteration,'-')
    plotLoss(testLoss, iteration,'--')
    plt.show()

    predictPlot(network=net, data=testData)
    result = predict(network=net, data=testData[:,:-1])
    print 1.0 * np.sum(result * testData[:,-1] > 0) / len(result)
    # for i in range(len(result)):
    #     print testData[i], result[i]

