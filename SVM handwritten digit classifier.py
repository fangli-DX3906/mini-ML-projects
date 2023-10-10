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


# prepare the raw data
wholeTrain = pd.read_csv('../../DigitsTraining.csv', header=None)
wholeTrain = np.asarray(wholeTrain)
wholeTest = pd.read_csv('../../DigitsTesting.csv', header=None)
wholeTest = np.asarray(wholeTest)
xraw_train = wholeTrain[:, 1:]
yraw_train = wholeTrain[:, 0]
xraw_test = wholeTest[:, 1:]
yraw_test = wholeTest[:, 0]

# one vs one case
targetDigit = 8
otherDigit = 1
pickTtr = yraw_train == targetDigit
pickOtr = yraw_train == otherDigit
pickTte = yraw_test == targetDigit
pickOte = yraw_test == otherDigit
xtrain1v1 = xraw_train[np.bitwise_or(pickTtr, pickOtr), :]
xtest1v1 = xraw_test[np.bitwise_or(pickTte, pickOte), :]
ytrain1v1 = yraw_train[np.bitwise_or(pickTtr, pickOtr)]
ytest1v1 = yraw_test[np.bitwise_or(pickTte, pickOte)]

# initialize a HandWrittenSVM instance and prepare the data set
HW1v1 = HandWrittenSVM(h=16, w=16, target=targetDigit)

xtrain1v1_resize = HW1v1.reSize(xtrain1v1)
xtest1v1_resize = HW1v1.reSize(xtest1v1)
Xtr1v1 = HW1v1.makeInput(xtrain1v1_resize, order=1)
Xte1v1 = HW1v1.makeInput(xtest1v1_resize, order=1)
Ytr1v1 = HW1v1.output2binary(ytrain1v1)
Yte1v1 = HW1v1.output2binary(ytest1v1)


e = []
t0 = time.time()
for c in range(1, 100, 1):
    sss = svm.SVC(C=c, kernel='linear')
    sss.fit(Xtr1v1, Ytr1v1)
    e.append(accuracy_score(Ytr1v1, sss.predict(Xtr1v1)))
Copt1v1 = range(1, 100, 1)[np.argmax(e)]
print(time.time() - t0)

# find out the optimal C
E1v1 = []
for c in range(1, 100, 1):
    sol1v1 = HW1v1.SVMSolver(C=c, X=Xtr1v1, Y=Ytr1v1)
    E1v1.append(HW1v1.countFalse(sol1v1[0], sol1v1[1], Xtr1v1, Ytr1v1))
plt.plot(range(1, 100, 1), E1v1)
plt.show()
Copt1v1 = np.argmin(E1v1)

sol1v1 = HW1v1.SVMSolver(C=Copt1v1, X=Xtr1v1, Y=Ytr1v1)
Ein1v1 = HW1v1.countFalse(sol1v1[0], sol1v1[1], Xtr1v1, Ytr1v1) / Xtr1v1.shape[0]
Eout1v1 = HW1v1.countFalse(sol1v1[0], sol1v1[1], Xte1v1, Yte1v1) / Xte1v1.shape[0]
HW1v1.plot2D(Xtr1v1, Ytr1v1, sol1v1[0], sol1v1[1], 'one-vs-one SVM classifier: training', ['r', 'b'])
HW1v1.plot2D(Xte1v1, Yte1v1, sol1v1[0], sol1v1[1], 'one-vs-one SVM classifier: testing', ['r', 'b'])

# one vs all case
targetDigit = 1

# # initialize a HandWrittenSVM instance and prepare the data set
HW1va = HandWrittenSVM(h=16, w=16, target=targetDigit)

xtrain1va_resize = HW1va.reSize(xraw_train)
xtest1va_resize = HW1va.reSize(xraw_test)
Xtr1va = HW1va.makeInput(xtrain1va_resize, order=1)
Xte1va = HW1va.makeInput(xtest1va_resize, order=1)
Ytr1va = HW1va.output2binary(yraw_train)
Yte1va = HW1va.output2binary(yraw_test)

# find out the optimal C
E1va = []
for c in range(1, 50, 1):
    sol1va = HW1va.SVMSolver(C=c, X=Xtr1va, Y=Ytr1va)
    E1va.append(HW1va.countFalse(sol1va[0], sol1va[1], Xtr1va, Ytr1va))
plt.plot(range(1, 50, 1), E1va)
plt.show()
Copt1va = np.argmin(E1va)

t0 = time.time()
sol1va = HW1va.SVMSolver(C=3, X=Xtr1va, Y=Ytr1va, show_progress=True)
print(time.time() - t0)
Ein1va = HW1va.countFalse(sol1va[0], sol1va[1], Xtr1va, Ytr1va) / Xtr1va.shape[0]
Eout1va = HW1va.countFalse(sol1va[0], sol1va[1], Xte1va, Yte1va) / Xte1va.shape[0]
HW1va.plot2D(Xtr1va, Ytr1va, sol1va[0], sol1va[1], 'one-vs-all SVM classifier: training', ['r', 'b'])
HW1va.plot2D(Xte1va, Yte1va, sol1va[0], sol1va[1], 'one-vs-all SVM classifier: testing', ['r', 'b'])
