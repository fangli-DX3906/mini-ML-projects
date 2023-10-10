import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt


# handwritten digit recognition with non a linear classifier

class HWNonLinearTrans:
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
            output.append([1 if numlist[i] == self.target else -1])
        return np.asarray(output).flatten()

    def countFalse(self, weights, input, output) -> tuple:
        X = np.concatenate((np.ones((input.shape[0], 1)), input), axis=1)
        w = np.array(weights).reshape((X.shape[1], 1))
        out = np.dot(X, w).reshape((X.shape[0],))
        temp = np.ones(X.shape[0])
        temp[np.where(out < 0)] = -1
        index = np.arange(0, X.shape[0])
        N = len(index[temp != output])
        falseIndex = index[temp != output].tolist()
        return N, falseIndex

    def gradCal(self, weight, x, y, d, batch=False) -> np.ndarray:
        w = weight.reshape((d, 1))
        if batch:
            xmat = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
            y = y.reshape((y.shape[0], 1))
        else:
            xmat = np.insert(x, 0, 1).reshape((1, d))
        pvec = np.dot(xmat.transpose(), y) / x.shape[0]
        Rmat = np.dot(xmat.transpose(), xmat) / x.shape[0]
        return (np.dot(Rmat, w) - pvec).reshape((d,))

    def linearReg(self, initw, input, output, maxIter, tol, eta, BGD=False) -> tuple:
        i = 0
        diff = 1
        weight = [initw]
        e = []
        perm = range(0, input.shape[0])
        while i < maxIter and diff > tol:
            error, index = self.countFalse(weight[i], input, output)
            e.append(error / input.shape[0])
            if BGD:
                gradient = self.gradCal(weight[i], input, output, weight[i].shape[0], batch=True)
            else:
                id = random.choice(perm)
                x = input[id, :]
                y = output[id]
                gradient = self.gradCal(weight[i], x, y, weight[i].shape[0], batch=False)
            weight.append(weight[i] - eta * gradient)
            diff = np.linalg.norm(weight[i + 1] - weight[i], ord=2)
            i += 1
        error_, index_ = self.countFalse(weight[-1], input, output)
        e.append(error_ / input.shape[0])
        return weight, e

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

    def plot2D(self, input, output, weight, title, colorStyle, order, plotLine=True):
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
            x = np.linspace(-1, 1, 1000)
            y = np.linspace(-1, 1, 1000)
            x, y = np.meshgrid(x, y)
            z = weight[0] + weight[1] * x + weight[2] * y
            index = 3
            for r in range(2, order + 1):
                for i in range(0, r + 1):
                    z += weight[index] * (x ** (r - i)) * (y ** i)
                    index += 1
            plt.contour(x, y, z, [0])
        plt.xlabel('Average Intensity')
        plt.ylabel('Average Symmetry')
        plt.title(title)
        plt.show()


# preset params
targetDigit = 1
initw1 = np.array([1e-1, 1e-1, 1e-1])
initw3 = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])

# data preprocessing
wholeTrain = pd.read_csv('../../DigitsTraining.csv', header=None)
wholeTrain = np.asarray(wholeTrain)
wholeTest = pd.read_csv('../../DigitsTesting.csv', header=None)
wholeTest = np.asarray(wholeTest)
xraw_train = wholeTrain[:, 1:]
yraw_train = wholeTrain[:, 0]
xraw_test = wholeTest[:, 1:]
yraw_test = wholeTest[:, 0]

# initialize an HWNonLinearTrans instance and prepare the data set
HW = HWNonLinearTrans(h=16, w=16, target=targetDigit)

# extract two features
xraw_train_resize = HW.reSize(xraw_train)
xraw_test_resize = HW.reSize(xraw_test)
ytrain = HW.output2binary(yraw_train)
ytest = HW.output2binary(yraw_test)

# 
xtrain = HW.makeInput(xraw_train_resize, order=1)
xtest = HW.makeInput(xraw_test_resize, order=1)

# searching for the optimal learning rate
lrB = []
rate = np.linspace(0, 1, 100)
t0 = time.time()
for r in rate:
    w1, errorB = HW.linearReg(initw1, xtrain, ytrain, eta=r, maxIter=1000, tol=1e-3, BGD=True)
    lrB.append(errorB[-1])
print(time.time() - t0)
# plot
plt.plot(range(100), lrB, c='r')
plt.title('Optimal Learning Rate for BGD')
plt.xlabel('Learning Rate')
plt.ylabel('Error')
plt.show()

b = np.argmin(np.array(lrB))
print(np.linspace(0, 1, 100)[b])  # 0.9797

# batch GD
Eout = []
w, Ein = HW.linearReg(initw1, xtrain, ytrain, eta=0.9797, maxIter=1000, tol=1e-3, BGD=True)
for i in w:
    num_out, index_out = HW.countFalse(i, xtest, ytest)
    Eout.append(num_out / xtest.shape[0])
HW.plot2D(xtrain, ytrain, w[-1], 'Linear Regression: Training Data (BGD)', ['r', 'b', 'k'], order=1)
HW.plot2D(xtest, ytest, w[-1], 'Linear Regression: Testing Data (BGD)', ['r', 'b', 'k'], order=1)
HW.compareError(Ein[200:], Eout[200:], ': Linear Regression (BGD)', ['r', 'b'])

# third order polynomial transformation
xtrain = HW.makeInput(xraw_train_resize, order=3)
xtest = HW.makeInput(xraw_test_resize, order=3)

# searching for the optimal learning rate
lrB = []
rate = np.linspace(0, 1, 100)
t0 = time.time()
for r in rate:
    w1, errorB = HW.linearReg(initw3, xtrain, ytrain, eta=r, maxIter=1000, tol=1e-3, BGD=True)
    lrB.append(errorB[-1])
print(time.time() - t0)
# plot
plt.plot(range(100), lrB, c='r')
plt.title('Optimal Learning Rate for BGD')
plt.xlabel('Learning Rate')
plt.ylabel('Error')
plt.show()

b = np.argmin(np.array(lrB))
print(np.linspace(0, 1, 100)[b])  

# batch GD
Eout = []
w, Ein = HW.linearReg(initw3, xtrain, ytrain, eta=0.9192, maxIter=8000, tol=1e-3, BGD=True)
for i in w:
    num_out, index_out = HW.countFalse(i, xtest, ytest)
    Eout.append(num_out / xtest.shape[0])
HW.plot2D(xtrain, ytrain, w[-1], 'Linear Regression: Training Data (BGD)', ['r', 'b', 'k'], order=3)
HW.plot2D(xtest, ytest, w[-1], 'Linear Regression: Testing Data (BGD)', ['r', 'b', 'k'], order=3)
HW.compareError(Ein[5000:], Eout[5000:], ': Linear Regression (BGD)', ['r', 'b'])