import numpy as np
from collections import Counter
import pandas as pd
import random
import matplotlib.pyplot as plt


# biary classification of Handwritten Digit Recognition 

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


# pre-set params
picSize = (16, 16)
pixelNum = 16 * 16
lastDig = 8
choice = 1
initw3 = np.array([0, 2, 1])
initw4 = np.array([3, 1, 2, 0])

#  import and pre-process data
wholeTrain = pd.read_csv('../../DigitsTraining.csv', header=None)
wholeTrain = np.asarray(wholeTrain)
wholeTest = pd.read_csv('../../DigitsTesting.csv', header=None)
wholeTest = np.asarray(wholeTest)
index_ch_tr = wholeTrain[:, 0] == choice
index_ld_tr = wholeTrain[:, 0] == lastDig
index_ch_te = wholeTest[:, 0] == choice
index_ld_te = wholeTest[:, 0] == lastDig
training_ = wholeTrain[np.bitwise_or(index_ch_tr, index_ld_tr), :]
testing_ = wholeTest[np.bitwise_or(index_ch_te, index_ld_te), :]

# initialize a instance of HandWritten class
hw1 = HandWritten(h=16, w=16, lstDgt=8, choice=1)

# feature extracting
training = hw1.reSize(training_, picSize)
testing = hw1.reSize(testing_, picSize)
x1_tr = np.asarray(hw1.avgSymmetry(training)).reshape((len(training), 1))
x2_tr = np.asarray(hw1.avgIntensity(training)).reshape((len(training), 1))
x1_te = np.asarray(hw1.avgSymmetry(testing)).reshape((len(testing), 1))
x2_te = np.asarray(hw1.avgIntensity(testing)).reshape((len(testing), 1))

# make final input matrix
x_train = np.concatenate((x1_tr, x2_tr), axis=1)
digit_train = training_[:, 0].astype(int)
x_test = np.concatenate((x1_te, x2_te), axis=1)
digit_test = testing_[:, 0].astype(int)
y_train = hw1.output2binary(digit_train)
y_test = hw1.output2binary(digit_test)

# 
Eout_a = []
w_a, Ein_a = hw1.perpcetron(initw3, x_train, y_train, eta=0.013, maxIter=200)
for i in w_a:
    num_out, index_out = hw1.countFalse(i, x_test, y_test)
    Eout_a.append(num_out / x_test.shape[0])
hw1.plot2D(x_train, digit_train, w_a[-1], 'PLA')
hw1.compareError(Ein_a, Eout_a, ': PLA')

# pocket algorithm
Eout_b = []
w_b, Ein_b = hw1.pocket(initw3, x_train, y_train, eta=0.013, maxIter=200)
for i in w_b:
    num_out, index_out = hw1.countFalse(i, x_test, y_test)
    Eout_b.append(num_out / x_test.shape[0])
hw1.plot2D(x_train, digit_train, w_b[-1], 'pocket PLA')
hw1.compareError(Ein_b, Eout_b, ': pocket PLA')

# linear regression
Eout_c = []
w_c, Ein_c = hw1.linearReg(initw3, x_train, y_train, maxIter=200, tolerance=1e-3, eta=0.013)
for i in w_c:
    num_out, index_out = hw1.countFalse(i, x_test, y_test)
    Eout_c.append(num_out / x_test.shape[0])
hw1.plot2D(x_train, digit_train, w_c[-1], 'Linear Regression')
hw1.compareError(Ein_c, Eout_c, ': Linear Regression')

# linear reg + pocket
Eout_d = []
w_d, Ein_d = hw1.pocket(w_c[-1], x_train, y_train, eta=0.02, maxIter=200)
for i in w_d:
    num_out, index_out = hw1.countFalse(i, x_test, y_test)
    Eout_d.append(num_out / x_test.shape[0])
hw1.plot2D(x_train, digit_train, w_d[-1], 'pocket PLA (LR weights)')
hw1.compareError(Ein_d, Eout_d, ': pocket PLA (LR weights)')

# one more features
x3_tr = np.asarray(hw1.maxWidth(training)).reshape((len(training), 1))
x3_te = np.asarray(hw1.maxWidth(testing)).reshape((len(testing), 1))
x_Train = np.concatenate((x_train, x3_tr), axis=1)
x_Test = np.concatenate((x_test, x3_te), axis=1)
mean1 = np.mean(x3_tr[digit_train == 1])
mean8 = np.mean(x3_tr[digit_train == 8])


# adding another feature
Eout_f1 = []
w_f1, Ein_f1 = hw1.perpcetron(initw4, x_Train, y_train, eta=0.002, maxIter=200)
for i in w_f1:
    num_out, index_out = hw1.countFalse(i, x_Test, y_test)
    Eout_f1.append(num_out / x_test.shape[0])
hw1.compareError(Ein_f1, Eout_f1, ': PLA (adding a new feature)')

Eout_f2 = []
w_f2, Ein_f2 = hw1.pocket(initw4, x_Train, y_train, eta=0.002, maxIter=200)
for i in w_f2:
    num_out, index_out = hw1.countFalse(i, x_Test, y_test)
    Eout_f2.append(num_out / x_test.shape[0])
hw1.compareError(Ein_f2, Eout_f2, ': pocket PLA (adding a new feature)')

Eout_f3 = []
w_f3, Ein_f3 = hw1.linearReg(initw4, x_Train, y_train, maxIter=200, tolerance=1e-3, eta=0.001)
for i in w_f3:
    num_out, index_out = hw1.countFalse(i, x_Test, y_test)
    Eout_f3.append(num_out / x_test.shape[0])
hw1.compareError(Ein_f3, Eout_f3, ': Linear Regression (adding a new feature)')

Eout_f4 = []
w_f4, Ein_f4 = hw1.pocket(w_f3[-1], x_Train, y_train, eta=0.001, maxIter=200)
for i in w_f4:
    num_out, index_out = hw1.countFalse(i, x_Test, y_test)
    Eout_f4.append(num_out / x_test.shape[0])
hw1.compareError(Ein_f4, Eout_f4, ': pocket PLA (adding a new feature)')
