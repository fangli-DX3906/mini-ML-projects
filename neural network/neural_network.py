import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import random
import time


class BackwardPropagation:
    def __init__(self, Ws, bs, L):
        self.biases = bs
        # transposed weights for each layers
        self.weights = Ws
        self.layers = L

    def tanh(self, s):
        return np.tanh(s)

    def tanhP(self, s):
        return 1 - np.tanh(s) ** 2

    def costP(self, xhat, y):
        return 2 * (xhat - y)

    def backwardProg(self, x, y):
        biasP = [np.zeros(bias.shape) for bias in self.biases]
        weightP = [np.zeros(weight.shape) for weight in self.weights]
        xVec = x
        xVecs = [x]
        sVecs = []
        delta = []
        for w, b in zip(self.weights, self.biases):
            xVec_temp = np.dot(w, xVec) + b
            sVecs.append(xVec_temp)
            xVec = self.tanh(xVec_temp)
            xVecs.append(xVec)
        # backward propagating
        delta.append(self.costP(xVecs[-1], y) * self.tanhP(sVecs[-1]))
        biasP[-1] = delta[0]
        weightP[-1] = np.dot(delta[0], xVecs[-2].transpose())
        for l in range(2, self.layers):
            sVec_temp = sVecs[-l]
            deriv = self.tanhP(sVec_temp)
            delta.append(np.dot(self.weights[-l + 1].transpose(), delta[l - 2]) * deriv)
            biasP[-l] = delta[-1]
            weightP[-l] = np.dot(delta[-1], xVecs[-l - 1].transpose())

        return weightP, biasP, delta, xVecs, sVecs



class NeutralNetwork:
    def __init__(self, dVec):
        self.L = len(dVec)
        self.dVec = dVec
        self.biases = [np.random.randn(d, 1) for d in self.dVec[1:]]
        self.weights = [np.random.randn(d, d_) for d_, d in zip(self.dVec[:-1], self.dVec[1:])]

    def tanh(self, s):
        return np.tanh(s)

    def tanhP(self, s):
        return 1 - (np.tanh(s)) ** 2

    def costP(self, xhat, y):
        return 2 * (xhat - y)

    def finalAcFunc(self, s):
        return s

    def finalAcFuncP(self):
        return 1.0

    def forwardProp(self, x):
        for w, b in zip(self.weights, self.biases):
            x = self.tanh(np.dot(x, w.T) + b.flatten())
        return x.flatten()

    def backwardProg(self, x, y):
        biasP = [np.zeros(bias.shape) for bias in self.biases]
        weightP = [np.zeros(weight.shape) for weight in self.weights]
        xVec = x
        xVecs = [x]
        sVecs = []
        delta = []
        ind = 0
        for w, b in zip(self.weights, self.biases):
            sVec_temp = np.dot(w, xVec).reshape((-1, 1)) + b
            sVecs.append(sVec_temp)
            if ind != self.L - 2:
                xVec = self.tanh(sVec_temp)
            else:
                xVec = self.finalAcFunc(sVec_temp)
            ind += 1
            xVecs.append(xVec)
        delta.append(self.costP(xVecs[-1], y) * self.finalAcFuncP())
        biasP[-1] = delta[0]
        weightP[-1] = np.dot(delta[0], xVecs[-2].transpose())
        for l in range(2, self.L):
            sVec_temp = sVecs[-l]
            deriv = self.tanhP(sVec_temp)
            delta.append(np.dot(self.weights[-l + 1].transpose(), delta[l - 2]) * deriv)
            biasP[-l] = delta[-1]
            weightP[-l] = np.dot(delta[-1], xVecs[-l - 1].reshape((1, -1)))
        return weightP, biasP, delta, xVecs, sVecs

    def batchUpdate(self, trainX, trainY):
        N = trainX.shape[0]
        biasP = [np.zeros(b.shape) for b in self.biases]
        weightP = [np.zeros(w.shape) for w in self.weights]
        for n in range(N):
            wPtemp, bPtemp, deltatemp, xVecstemp, sVecstemp = self.backwardProg(trainX[n], trainY[n])
            biasP = [bP + bPdelta / N for bP, bPdelta in zip(biasP, bPtemp)]
            weightP = [wP + wPdelta / N for wP, wPdelta in zip(weightP, wPtemp)]
        return weightP, biasP

    def stochasticUpdate(self, trainX, trainY):
        N = trainX.shape[0]
        candidate = [n for n in range(N)]
        random.shuffle(candidate)
        pick = candidate[0]
        wPtemp, bPtemp, deltatemp, xVecstemp, sVecstemp = self.backwardProg(trainX[pick], trainY[pick])
        weightP = wPtemp
        biasP = bPtemp
        return weightP, biasP

    def gradientDescent(self, trainX, trainY, testX, testY, maxIter, eta, SGD=False):
        Ein = []
        Eout = []
        for iter in range(maxIter):
            if SGD:
                weightP, biasP = self.stochasticUpdate(trainX, trainY)
            else:
                weightP, biasP = self.batchUpdate(trainX, trainY)
            self.biases = [b - eta * bP for b, bP in zip(self.biases, biasP)]
            self.weights = [w - eta * wP for w, wP in zip(self.weights, weightP)]
            print('This is the {}th iteration'.format(iter))
            Ein.append(np.sum(np.sign(self.forwardProp(trainX)) != trainY) / trainX.shape[0])
            Eout.append(np.sum(np.sign(self.forwardProp(testX)) != testY) / testX.shape[0])
        return self.weights, self.biases, Ein, Eout


class HWDataPreProceesing:
    def __init__(self, h, w, target):
        self.height = h
        self.width = w
        self.pixelNum = h * w
        self.picSize = (self.width, self.height)
        self.target = target

    def avgSymmetry(self, data) -> np.ndarray:
        avgsyty = []
        for i in range(len(data)):
            vSym = np.sum(np.abs(data[i] - data[i][::-1, :])) / self.pixelNum
            hSym = np.sum(np.abs(data[i] - data[i][:, ::-1])) / self.pixelNum
            avgsyty.append(-0.5 * (vSym + hSym))
        return np.asarray(avgsyty)

    def avgIntensity(self, data) -> np.ndarray:
        avgitsy = []
        for i in range(len(data)):
            avgitsy.append(-np.sum(data[i]) / self.pixelNum)
        return np.asarray(avgitsy)

    def reSize(self, data) -> list:
        picResize = []
        for i in range(data.shape[0]):
            picResize.append(data[i, :].reshape(self.picSize))
        return picResize

    def makeInput(self, data, order) -> np.ndarray:
        intensity = self.avgIntensity(data).reshape((len(data), 1))
        symmetry = self.avgSymmetry(data).reshape((len(data), 1))
        outputdata = np.concatenate((intensity, symmetry), axis=1)
        if order >= 2:
            for r in range(2, order + 1):
                for i in range(0, r + 1):
                    temp = intensity ** (r - i) * symmetry ** i
                    outputdata = np.concatenate((outputdata, temp), axis=1)
        return outputdata

    def output2binary(self, numlist) -> np.ndarray:
        output = []
        for i in range(numlist.shape[0]):
            output.append([1. if numlist[i] == self.target else -1.])
        return np.asarray(output).flatten()
