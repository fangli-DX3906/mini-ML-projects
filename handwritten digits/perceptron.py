import numpy as np
from collections import Counter
import pandas as pd
import random
import matplotlib.pyplot as plt


# define a class
class HandWritten:
    def __init__(self, h, w, lstDgt, choice):
        self.height = h
        self.width = w
        self.pixelNum = h * w
        self.lstDgt = lstDgt
        self.choice = choice

    def avgSymmetry(self, data) -> list:
        avgsyty = []
        for i in range(len(data)):
            vSym = np.sum(np.abs(data[i] - data[i][::-1, :])) / self.pixelNum
            hSym = np.sum(np.abs(data[i] - data[i][:, ::-1])) / self.pixelNum
            avgsyty.append(-0.5 * (vSym + hSym))
        return avgsyty

    def avgIntensity(self, data) -> list:
        avgitsy = []
        for i in range(len(data)):
            avgitsy.append(np.sum(data[i]) / self.pixelNum)
        return avgitsy

    def maxWidth(self, data) -> list:
        width = []
        for i in range(len(data)):
            pressdown = np.sum(data[i], axis=0)
            index = np.where(pressdown != 16)[0]
            width.append(index.shape[0])
        return width

    def reSize(self, data, size):
        result = []
        for i in range(data.shape[0]):
            result.append(data[i, 1:].reshape(size))
        return result

    def output2binary(self, input):
        output = []
        for i in range(input.shape[0]):
            output.append([1 if input[i] == self.choice else -1])
        return np.asarray(output).flatten()

    def countFalse(self, weights, input, output):
        X = np.concatenate((np.ones((input.shape[0], 1)), input), axis=1)
        w = np.array(weights).reshape((X.shape[1], 1))
        out = np.dot(X, w).reshape((X.shape[0],))
        temp = np.ones(X.shape[0])
        temp[np.where(out < 0)] = -1
        index = np.arange(0, X.shape[0])
        N = len(index[temp != output])
        falseIndex = index[temp != output].tolist()
        return N, falseIndex

    def compareError(self, errorIn, errorOut, title):
        num = len(errorIn)
        plt.figure()
        plt.plot(range(num), errorIn, color='r')
        plt.plot(range(num), errorOut, color='b')
        plt.legend(['Ein', 'Eout'])
        plt.xlabel('iteration')
        plt.ylabel('error')
        plt.title('In sample and Out of Sample Comparison' + title)
        plt.show()

    def plot2D(self, input, output, weights, title):
        colorDict = {self.choice: 'r', self.lstDgt: 'b'}
        colorList = list(map(lambda val: colorDict[val], list(output)))
        plt.figure()
        plt.scatter(input[:, 0], input[:, 1], c=colorList)
        xrange = np.max(input[:, 0]) - np.min(input[:, 0])
        yrange = np.max(input[:, 1]) - np.min(input[:, 1])
        edgelen = max(xrange, yrange)
        xlim = [np.min(input[:, 0]), np.min(input[:, 0]) + edgelen]
        ylim = [np.min(input[:, 1]), np.min(input[:, 1]) + edgelen]
        plt.xlim(xlim)
        plt.ylim(ylim)
        x = np.linspace(-1, 1, 100)
        y = -weights[0] / weights[2] - weights[1] / weights[2] * x
        plt.plot(x, y, c='k')
        plt.xlabel('Average Symmetry')
        plt.ylabel('Average Intensity')
        plt.title(title)
        plt.show()

    def perpcetron(self, initialWeight, input, output, eta, maxIter):
        weights = [initialWeight]
        e = []
        for i in range(maxIter):
            error, index = self.countFalse(weights[i], input, output)
            if error == 0:
                break
            e.append(error / input.shape[0])
            random.shuffle(index)
            inputVec = np.insert(input[index[0]], 0, 1)
            weights.append(weights[i] + eta * inputVec * output[index[0]])
        error_, index_ = self.countFalse(weights[-1], input, output)
        e.append(error_ / input.shape[0])
        return weights, e

    def pocket(self, initialWeight, input, output, eta, maxIter):
        weights = initialWeight
        e = []
        error_old, index = self.countFalse(weights, input, output)
        bestweights = weights
        bw = [bestweights]
        e.append(error_old / input.shape[0])
        for i in range(maxIter):
            if error_old == 0:
                break
            random.shuffle(index)
            inputVec = np.insert(input[index[0]], 0, 1)
            weights = weights + eta * inputVec * output[index[0]]
            error_now, index = self.countFalse(weights, input, output)
            if error_now <= error_old:
                bestweights = weights.copy()
                error_old = error_now
            bw.append(bestweights)
            e.append(error_old / input.shape[0])
        return bw, e

    def gradientCal(self, weight, xval, yval, d):
        w = weight.reshape((d, 1))
        xvec = xval.reshape((d, 1))
        Rmat = np.dot(xvec, xvec.transpose())
        pvec = yval * xvec
        return (2 * (np.dot(Rmat, w) - pvec)).reshape((d,))

    def linearReg(self, initialWeight, input, output, maxIter, tolerance, eta):
        i = 0
        diff = 1
        weight = [initialWeight]
        e = []
        while i < maxIter and diff > tolerance:
            error, index = self.countFalse(weight[i], input, output)
            e.append(error / input.shape[0])
            random.shuffle(index)
            inputVec = np.insert(input[index[0]], 0, 1)
            gradient = self.gradientCal(weight[i], inputVec, output[index[0]], inputVec.shape[0])
            weight.append(weight[i] - eta * gradient)
            diff = np.linalg.norm(weight[i + 1] - weight[i], ord=2)
            i += 1
        error_, index_ = self.countFalse(weight[-1], input, output)
        e.append(error_ / input.shape[0])
        return weight, e

    def linearRegPocket(self, initialWeight, input, output, maxIter, tolerance, eta):
        error_old, index = self.countFalse(initialWeight, input, output)
        weights = initialWeight
        bestweights = weights
        repeat = 0
        diff = 1
        while repeat < maxIter and diff > tolerance:
            if error_old == 0:
                break
            inputVec = np.insert(input[index[0]], 0, 1)
            gradient = self.gradientCal(weights, inputVec, output[index[0]], inputVec.shape[0])
            weights_new = weights - eta * gradient
            error_now, index = self.countFalse(weights_new, input, output)
            if error_now <= error_old:
                bestweights = weights_new.copy()
                error_old = error_now
            diff = np.linalg.norm(weights_new - weights, ord=2)
            weights = weights_new
            repeat += 1
        return bestweights
