import numpy as np
import scipy.io
from scipy import stats
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image


# perceptron on a fake dataset


def output2binary(w, input):
    output = []
    for i in range(input.shape[0]):
        output.append([1 if w[0] + w[1] * input[i, 0] + w[2] * input[i, 1] >= 0 else -1])
    return np.asarray(output).flatten()


def plotPLA(w, input, output, lim, colorStyle, new=True, show=True):
    colorDict = {1: colorStyle[0], -1: colorStyle[1]}
    colorList = list(map(lambda x: colorDict[x], list(output)))
    if new:
        plt.figure()
    plt.xlim(lim)
    plt.ylim(lim)
    plt.scatter(input[:, 0], input[:, 1], c=colorList)
    x1 = np.linspace(-2.5, 2.5, 50)
    x2 = -w[0] / w[2] - w[1] / w[2] * x1
    plt.plot(x1, x2, c=colorStyle[2])
    plt.xlabel('x1')
    plt.ylabel('x2')
    if show:
        plt.show()


def countFalse(weights, input, output):
    N = 0
    falseIndex = []
    for i in range(input.shape[0]):
        xVec = np.insert(input[i], 0, 1)
        outputHat = [1 if np.dot(weights, xVec) >= 0 else -1]
        if output[i] != outputHat:
            N += 1
            falseIndex.append(i)
    return N, falseIndex


def perpcetron(initialWeight, input, output, maxIter=500, printWeight=False):
    N = 1
    weights = [initialWeight]
    repeat = 0
    while N != 0 and repeat <= maxIter:
        N, index = countFalse(weights[repeat], input, output)
        if N == 0:
            break
        repeat += 1
        random.shuffle(index)
        inputVec = np.insert(input[index[0]], 0, 1)
        weights.append(weights[repeat - 1] + inputVec * output[index[0]])
        if printWeight:
            print('-------------> This is the {} iteration <-------------'.format(repeat))
            print('Intercept is {}, two slopes are {} and {}'.format(weights[-1][0], weights[-1][1], weights[-1][2]))
    return weights


# pre-set parameters
initWeights = np.array([1, 0, 1])
targetWeights = np.array([1, 5, 2])
colorStyle1 = ['b', 'r', 'k']
colorStyleN = ['b', 'r', 'g']

# on inputData
inputData = pd.read_csv('../../inputData.csv')
inputData = np.asarray(inputData)
outputData = output2binary(targetWeights, inputData)
itvl = [-2.5, 2.5]
plotPLA(targetWeights, inputData, outputData, itvl, colorStyle1, new=True)
w = perpcetron(initWeights, inputData, outputData, maxIter=100, printWeight=True)
plotPLA(targetWeights, inputData, outputData, itvl, colorStyle1, new=True, show=False)
plotPLA(w[-1], inputData, outputData, itvl, colorStyleN, new=False, show=True)

# on inputData20
inputData20 = pd.read_csv('../../inputData20.csv')
inputData20 = np.asarray(inputData20)
outputData20 = output2binary(targetWeights, inputData20)
itvl20 = [-3, 3]
w20 = perpcetron(initWeights, inputData20, outputData20, maxIter=100, printWeight=True)
plotPLA(targetWeights, inputData20, outputData20, itvl20, colorStyle1, new=True, show=False)
plotPLA(w20[-1], inputData20, outputData20, itvl20, colorStyleN, new=False, show=True)

# on inputData100
inputData100 = pd.read_csv('../../inputData100.csv')
inputData100 = np.asarray(inputData100)
outputData100 = output2binary(targetWeights, inputData100)
itvl100 = [-3, 3]
w100 = perpcetron(initWeights, inputData100, outputData100, maxIter=100, printWeight=True)
plotPLA(targetWeights, inputData100, outputData100, itvl100, colorStyle1, new=True, show=False)
plotPLA(w100[-1], inputData100, outputData100, itvl100, colorStyleN, new=False, show=True)

# on inputData1000
inputData1000 = pd.read_csv('../../inputData1000.csv')
inputData1000 = np.asarray(inputData1000)
outputData1000 = output2binary(targetWeights, inputData1000)
itvl1000 = [-4, 4]
w1000 = perpcetron(initWeights, inputData1000, outputData1000, maxIter=300, printWeight=True)
plotPLA(targetWeights, inputData1000, outputData1000, itvl1000, colorStyle1, new=True, show=False)
plotPLA(w1000[-1], inputData1000, outputData1000, itvl1000, colorStyleN, new=False, show=True)