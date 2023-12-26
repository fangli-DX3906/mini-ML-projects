import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from cvxopt import matrix, solvers

# preset the params
class HandWrittenSVM:
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

    def SVMSolver(self, C, X, Y, show_progress=False) -> tuple:
        # using cvxopt for quadratic programming
        num = X.shape[0]
        ones = np.ones(num).astype('float64')
        zeros = np.zeros(num).astype('float64')
        kernel = np.dot(X, np.transpose(X)) + np.diag(ones) * (1 / C)
        P = matrix(kernel * np.outer(Y, Y))
        q = matrix(-ones.reshape((-1, 1)))
        G = matrix(-np.eye(num))
        h = matrix(zeros.reshape((-1, 1)))
        A = matrix(Y, (1, num))
        b = matrix(0.)
        solvers.options['show_progress'] = show_progress
        sol = solvers.qp(P, q, G, h, A, b)
        alpha = np.array(sol['x'])
        alpha[alpha <= 1e-4] = 0
        alpha.astype(np.float32)
        # calculate w and b
        w = np.sum(alpha * Y.reshape((-1, 1)) * X, axis=0)
        picking = alpha.flatten() != 0
        b = np.mean(Y[picking] - np.dot(w, X[picking, :].T))
        return w, b
        pass

    def updateFunc(self, i, j, C, inputData):
        # l = 0
        # h = C
        # if inputData[i][-1] == inputData[j][-1]:
        pass

    def countFalse(self, w, b, X, Y):
        return np.sum(np.sign(Y) != np.sign(np.dot(w, X.T) + b))

    def compareError(self, errorIn, errorOut, title, colorList):
        num = len(errorIn)
        plt.figure()
        plt.plot(range(num), errorIn, color=colorList[0], linewidth=0.5)
        plt.plot(range(num), errorOut, color=colorList[1], linewidth=0.5)
        plt.legend(['Ein', 'Eout'])
        plt.xlabel('iteration')
        plt.ylabel('error')
        plt.title('In sample and Out of Sample Comparison' + title)
        plt.show()

    def plot2D(self, input, output, w, b, title, colorStyle, plotLine=True):
        colorDict = {1: colorStyle[0], -1: colorStyle[1]}
        colorList = list(map(lambda val: colorDict[val], list(output)))
        plt.figure()
        plt.scatter(input[:, 0], input[:, 1], c=colorList, s=1)
        xrange = np.max(input[:, 0]) - np.min(input[:, 0])
        yrange = np.max(input[:, 1]) - np.min(input[:, 1])
        edgelen = max(xrange, yrange)
        xlim = [np.min(input[:, 0]), np.min(input[:, 0]) + edgelen]
        ylim = [np.min(input[:, 1]), np.min(input[:, 1]) + edgelen]
        plt.xlim(xlim)
        plt.ylim(ylim)
        if plotLine:
            x = np.linspace(np.min(input[:, 0]), np.min(input[:, 0]) + edgelen, 1000)
            slope = -w[0] / w[1]
            intercept = -b / w[1]
            y = intercept + slope * x
            plt.plot(x, y, 'k-')
        plt.xlabel('Average Intensity')
        plt.ylabel('Average Symmetry')
        plt.title(title)
        plt.show()